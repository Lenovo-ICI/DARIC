#include "cache_ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <algorithm>

namespace {

struct StridedDimension {
    int dim;
    int64_t size;
    int64_t stride;
};

long long checked_multiply(long long lhs, long long rhs, const char* context) {
    TORCH_CHECK(lhs >= 0 && rhs >= 0, context, " must be non-negative");
    TORCH_CHECK(lhs == 0 || rhs <= std::numeric_limits<long long>::max() / lhs,
                context, " exceeds int64 range");
    return lhs * rhs;
}

long long get_dense_span_elements(const torch::Tensor& tensor, int first_dim) {
    std::vector<StridedDimension> dimensions;
    dimensions.reserve(static_cast<size_t>(tensor.dim() - first_dim));

    for (int d = first_dim; d < tensor.dim(); ++d) {
        const int64_t size = tensor.size(d);
        const int64_t stride = tensor.stride(d);
        TORCH_CHECK(size > 0,
                    "KV cache copy dimensions must be non-empty"
                    " | shape=", tensor.sizes(),
                    " | stride=", tensor.strides(),
                    " | failed_dim=", d);
        if (size == 1) continue;
        TORCH_CHECK(stride > 0,
                    "KV cache non-singleton dimensions must have positive strides"
                    " | shape=", tensor.sizes(),
                    " | stride=", tensor.strides(),
                    " | failed_dim=", d,
                    " | actual_stride=", stride);
        dimensions.push_back({d, size, stride});
    }

    std::sort(dimensions.begin(), dimensions.end(),
              [](const StridedDimension& lhs, const StridedDimension& rhs) {
                  return lhs.stride < rhs.stride;
              });

    long long expected_stride = 1;
    for (const auto& dimension : dimensions) {
        TORCH_CHECK(dimension.stride == expected_stride,
                    "KV cache copy dimensions must form a dense physical span"
                    " | shape=", tensor.sizes(),
                    " | stride=", tensor.strides(),
                    " | failed_dim=", dimension.dim,
                    " | expected_stride=", expected_stride,
                    " | actual_stride=", dimension.stride,
                    " | relation=",
                    dimension.stride > expected_stride ? "hole" : "overlap");
        expected_stride = checked_multiply(
            expected_stride, dimension.size, "KV cache dense span");
    }
    return expected_stride;
}

void validate_outer_regions(
    const torch::Tensor& tensor, int prefix_size, int num_pages,
    long long prefix_stride, long long page_stride, long long suffix_size
) {
    std::vector<std::pair<long long, long long>> axes;
    if (prefix_size > 1) axes.emplace_back(prefix_stride, prefix_size);
    if (num_pages > 1) axes.emplace_back(page_stride, num_pages);
    std::sort(axes.begin(), axes.end());

    long long occupied_span = suffix_size;
    for (const auto& [stride, size] : axes) {
        TORCH_CHECK(stride >= occupied_span,
                    "KV cache prefix/page regions overlap"
                    " | shape=", tensor.sizes(),
                    " | stride=", tensor.strides(),
                    " | region_size=", suffix_size,
                    " | required_stride=", occupied_span,
                    " | actual_stride=", stride);
        const long long axis_extent = checked_multiply(
            size - 1, stride, "KV cache outer span");
        TORCH_CHECK(axis_extent <= std::numeric_limits<long long>::max() - occupied_span,
                    "KV cache outer span exceeds int64 range");
        occupied_span += axis_extent;
    }
}

}  // namespace

void get_page_copy_shape(
    const torch::Tensor& tensor, bool is_page_first,
    int& num_pages, int& prefix_size, long long& page_stride,
    long long& suffix_size, long long& prefix_stride
) {
    TORCH_CHECK(tensor.is_cuda(), "KV cache tensor must be on CUDA");
    TORCH_CHECK(tensor.dim() >= 1, "KV cache tensor must have at least one dimension");
    const long long element_size = tensor.element_size();

    if (is_page_first) {
        TORCH_CHECK(tensor.size(0) > 0, "page dimension must be non-zero");
        TORCH_CHECK(tensor.size(0) <= std::numeric_limits<int>::max(),
                    "number of pages exceeds int range: ", tensor.size(0));

        num_pages = static_cast<int>(tensor.size(0));
        prefix_size = 1;
        prefix_stride = 0;
        page_stride = checked_multiply(
            tensor.stride(0), element_size, "KV cache page stride");
        suffix_size = checked_multiply(
            get_dense_span_elements(tensor, 1), element_size,
            "KV cache page span");
        if (num_pages > 1) {
            TORCH_CHECK(page_stride >= suffix_size,
                        "KV cache pages overlap"
                        " | shape=", tensor.sizes(),
                        " | stride=", tensor.strides(),
                        " | page_stride=", page_stride,
                        " | page_span=", suffix_size);
        }
    } else {
        TORCH_CHECK(tensor.dim() >= 2, "non-page-first KV cache tensor must have at least two dimensions");
        TORCH_CHECK(tensor.size(0) > 0, "prefix dimension must be non-zero");
        TORCH_CHECK(tensor.size(1) > 0, "page dimension must be non-zero");

        TORCH_CHECK(tensor.size(0) <= std::numeric_limits<int>::max(),
                    "prefix size exceeds int range: ", tensor.size(0));
        TORCH_CHECK(tensor.size(1) <= std::numeric_limits<int>::max(),
                    "number of pages exceeds int range: ", tensor.size(1));
        prefix_size = static_cast<int>(tensor.size(0));
        num_pages = static_cast<int>(tensor.size(1));
        prefix_stride = checked_multiply(
            tensor.stride(0), element_size, "KV cache prefix stride");
        page_stride = checked_multiply(
            tensor.stride(1), element_size, "KV cache page stride");
        suffix_size = checked_multiply(
            get_dense_span_elements(tensor, 2), element_size,
            "KV cache suffix span");
        validate_outer_regions(
            tensor, prefix_size, num_pages,
            prefix_stride, page_stride, suffix_size);
    }

    TORCH_CHECK(page_stride > 0, "page_stride must be positive");
}

void validate_common_job_inputs(
    size_t num_jobs, const std::vector<int64_t>& offsets,
    const std::vector<int64_t>& byte_sizes,
    const std::vector<int64_t>& indices_offsets,
    const torch::Tensor& indices_flat
) {
    TORCH_CHECK(offsets.size() == num_jobs, "offsets length mismatch");
    TORCH_CHECK(byte_sizes.size() == num_jobs, "byte_sizes length mismatch");
    TORCH_CHECK(indices_offsets.size() == num_jobs + 1, "indices_offsets size must be num_jobs + 1");

    TORCH_CHECK(indices_flat.is_cuda(), "indices_flat must be on CUDA");
    TORCH_CHECK(indices_flat.dim() == 1, "indices_flat must be 1D");
    TORCH_CHECK(indices_flat.dtype() == torch::kInt32, "indices_flat must be int32");
    TORCH_CHECK(indices_flat.is_contiguous(), "indices_flat must be contiguous");

    TORCH_CHECK(indices_offsets.front() >= 0, "indices_offsets[0] must be non-negative");
    TORCH_CHECK(indices_offsets.back() <= indices_flat.numel(), "indices_offsets.back() exceeds indices_flat.numel()");

    for (size_t i = 0; i < num_jobs; ++i) {
        TORCH_CHECK(offsets[i] >= 0, "negative store offset at job ", i);
        TORCH_CHECK(byte_sizes[i] >= 0, "negative byte size at job ", i);
        TORCH_CHECK(indices_offsets[i] >= 0, "negative indices offset at job ", i);
        TORCH_CHECK(indices_offsets[i] <= indices_offsets[i + 1], "indices_offsets must be non-decreasing at job ", i);
    }
}

cudaDeviceProp get_device_properties(int device_index) {
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_index));
    return prop;
}

void record_tensor_on_external_stream(const torch::Tensor& tensor, cudaStream_t stream) {
    const auto external_stream = c10::cuda::getStreamFromExternal(stream, tensor.get_device());
    c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), external_stream);
}

void batched_memcpy_d2h(
    torch::Tensor staging,
    torch::Tensor shm,
    const std::vector<int64_t>& stg_offsets,
    const std::vector<int64_t>& shm_offsets,
    const std::vector<int64_t>& sizes,
    uintptr_t stream_ptr
) {
    // Issue one copy-engine cudaMemcpyAsync per (src, dst, size) region on the
    // given stream. The CPU `shm` must be page-locked (cudaHostRegister) so the
    // copy engine is used and the DMA overlaps with SM work on other streams.
    // This bypasses PyTorch's is_pinned() check (which is False for
    // cudaHostRegister'd-frombuffer tensors) by calling cudaMemcpyAsync directly.
    const size_t n = sizes.size();
    if (n == 0) return;
    cudaStream_t stream = (stream_ptr != 0)
        ? reinterpret_cast<cudaStream_t>(stream_ptr)
        : at::cuda::getCurrentCUDAStream();
    const uint8_t* stg = staging.data_ptr<uint8_t>();
    uint8_t* sh = shm.data_ptr<uint8_t>();
    {
        py::gil_scoped_release release;
        for (size_t i = 0; i < n; ++i) {
            CUDA_CHECK(cudaMemcpyAsync(
                sh + shm_offsets[i],
                stg + stg_offsets[i],
                static_cast<size_t>(sizes[i]),
                cudaMemcpyDeviceToHost, stream));
        }
    }
}

void batched_memcpy_h2d(
    torch::Tensor shm,
    torch::Tensor staging,
    const std::vector<int64_t>& shm_offsets,
    const std::vector<int64_t>& stg_offsets,
    const std::vector<int64_t>& sizes,
    uintptr_t stream_ptr
) {
    // Mirror of batched_memcpy_d2h for the staged read path: copy-engine DMA
    // from page-locked shm into a GPU staging buffer, after which the scatter
    // kernel reads GPU memory only. Needed on platforms where SM zero-copy
    // reads of registered host memory fault (e.g. Hygon DCU / DTK).
    const size_t n = sizes.size();
    if (n == 0) return;
    cudaStream_t stream = (stream_ptr != 0)
        ? reinterpret_cast<cudaStream_t>(stream_ptr)
        : at::cuda::getCurrentCUDAStream();
    const uint8_t* sh = shm.data_ptr<uint8_t>();
    uint8_t* stg = staging.data_ptr<uint8_t>();
    {
        py::gil_scoped_release release;
        for (size_t i = 0; i < n; ++i) {
            CUDA_CHECK(cudaMemcpyAsync(
                stg + stg_offsets[i],
                sh + shm_offsets[i],
                static_cast<size_t>(sizes[i]),
                cudaMemcpyHostToDevice, stream));
        }
    }
}
