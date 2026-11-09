#include "cache_ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>

// ==================== KERNELS ====================
__global__ void fused_gather_pages_kernel(const PageCopyJobDesc* __restrict__ jobs, int num_jobs) {
    const int job = static_cast<int>(blockIdx.y);
    if (job >= num_jobs) return;

    const PageCopyJobDesc& j = jobs[job];
    const long long within_idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (within_idx >= j.num_elements) return;

    // 直接按字节操作
    const char* src = reinterpret_cast<const char*>(j.src);
    char* dst = reinterpret_cast<char*>(j.dst);
    const int* indices = j.indices;

    if (j.is_page_first) {
        const long long page_local = within_idx / j.suffix_size;
        const long long offset_in_page = within_idx % j.suffix_size;
        const int src_page = indices[page_local];
        
        // 【关键修改】跳过 block_id 为 0 的页，不读取 GPU 显存，也不写入 SHM
        if (src_page == 0) return; 
        
        const long long src_idx = static_cast<long long>(src_page) * j.page_stride + offset_in_page;
        dst[within_idx] = src[src_idx];
    } else {
        const long long suffix_idx = within_idx % j.suffix_size;
        const long long temp = within_idx / j.suffix_size;
        const int page_local = static_cast<int>(temp % j.num_selected_pages);
        const int prefix_idx = static_cast<int>(temp / j.num_selected_pages);

        const int src_page = indices[page_local];
        
        // 【关键修改】跳过 block_id 为 0 的页
        if (src_page == 0) return;

        const long long src_idx = static_cast<long long>(prefix_idx) * j.prefix_stride +
                                  static_cast<long long>(src_page) * j.page_stride + suffix_idx;
        dst[within_idx] = src[src_idx];
    }
}

// 16-bytes-per-thread variant: far fewer threads and full-width memory
// transactions (the 1-byte/thread kernel above underutilizes bandwidth,
// especially for GPU->pinned-host writes). Requires every job to be 16-byte
// aligned (suffix_size, page_stride, prefix_stride, num_elements all % 16 == 0);
// the launcher falls back to the byte kernel otherwise.
__global__ void fused_gather_pages_kernel_int4(const PageCopyJobDesc* __restrict__ jobs, int num_jobs) {
    const int job = static_cast<int>(blockIdx.y);
    if (job >= num_jobs) return;

    const PageCopyJobDesc& j = jobs[job];
    const long long within16 = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; // 16-byte unit
    const long long byte_idx = within16 * 16;
    if (byte_idx >= j.num_elements) return;

    const char* src = reinterpret_cast<const char*>(j.src);
    char* dst = reinterpret_cast<char*>(j.dst);
    const int* indices = j.indices;

    if (j.is_page_first) {
        const long long page_local = byte_idx / j.suffix_size;
        const long long offset_in_page = byte_idx % j.suffix_size;
        const int src_page = indices[page_local];
        if (src_page == 0) return;
        const long long src_idx = static_cast<long long>(src_page) * j.page_stride + offset_in_page;
        *reinterpret_cast<int4*>(dst + byte_idx) = *reinterpret_cast<const int4*>(src + src_idx);
    } else {
        const long long suffix_idx = byte_idx % j.suffix_size;
        const long long temp = byte_idx / j.suffix_size;
        const int page_local = static_cast<int>(temp % j.num_selected_pages);
        const int prefix_idx = static_cast<int>(temp / j.num_selected_pages);

        const int src_page = indices[page_local];
        if (src_page == 0) return;
        const long long src_idx = static_cast<long long>(prefix_idx) * j.prefix_stride +
                                  static_cast<long long>(src_page) * j.page_stride + suffix_idx;
        *reinterpret_cast<int4*>(dst + byte_idx) = *reinterpret_cast<const int4*>(src + src_idx);
    }
}

__global__ void fused_scatter_pages_kernel(const PageCopyJobDesc* __restrict__ jobs, int num_jobs) {
    const int job = static_cast<int>(blockIdx.y);
    if (job >= num_jobs) return;

    const PageCopyJobDesc& j = jobs[job];
    const long long within_idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (within_idx >= j.num_elements) return;

    const char* src = reinterpret_cast<const char*>(j.src);
    char* dst = reinterpret_cast<char*>(j.dst);
    const int* indices = j.indices;

    if (j.is_page_first) {
        const long long page_local = within_idx / j.suffix_size;
        const long long offset_in_page = within_idx % j.suffix_size;
        const int dst_page = indices[page_local];
        
        // 【关键修改】跳过 block_id 为 0 的页，不读取 SHM，也不写入 GPU 显存
        if (dst_page == 0) return; 
        
        const long long dst_idx = static_cast<long long>(dst_page) * j.page_stride + offset_in_page;
        dst[dst_idx] = src[within_idx];
    } else {
        const long long suffix_idx = within_idx % j.suffix_size;
        const long long temp = within_idx / j.suffix_size;
        const int page_local = static_cast<int>(temp % j.num_selected_pages);
        const int prefix_idx = static_cast<int>(temp / j.num_selected_pages);

        const int dst_page = indices[page_local];
        
        // 【关键修改】跳过 block_id 为 0 的页
        if (dst_page == 0) return;

        const long long dst_idx = static_cast<long long>(prefix_idx) * j.prefix_stride +
                                  static_cast<long long>(dst_page) * j.page_stride + suffix_idx;
        dst[dst_idx] = src[within_idx];
    }
}

// ==================== LAUNCHERS ====================

static void fused_gather_pages_launch(
    const std::vector<torch::Tensor>& srcs, uint8_t* store_base_ptr,
    const std::vector<int64_t>& offsets, const std::vector<int64_t>& byte_sizes,
    const int* flat_indices_ptr, const std::vector<int64_t>& indices_offsets,
    const std::vector<char>& is_page_first_list, cudaStream_t stream
) {
    const size_t num_jobs = srcs.size();
    if (num_jobs == 0) return;

    std::vector<PageCopyJobDesc> h_jobs(num_jobs);
    long long max_elements = 0;

    for (size_t i = 0; i < num_jobs; ++i) {
        const auto& src = srcs[i];
        const bool is_pf = static_cast<bool>(is_page_first_list[i]);

        int num_pages = 0, prefix_size = 0;
        long long page_stride = 0, suffix_size = 0, prefix_stride = 0;
        get_page_copy_shape(src, is_pf, num_pages, prefix_size, page_stride, suffix_size, prefix_stride);

        const int64_t num_sel_i64 = indices_offsets[i + 1] - indices_offsets[i];
        const int num_sel = static_cast<int>(num_sel_i64);

        // 此时 num_elem 直接就是字节数
        const long long num_elem = static_cast<long long>(prefix_size) * num_sel * suffix_size;

        if (static_cast<long long>(byte_sizes[i]) != num_elem) {
            std::stringstream ss;
            ss << "byte_size mismatch at job " << i 
               << " | C++ expected bytes: " << num_elem 
               << " | Python passed bytes: " << byte_sizes[i]
               << " | Tensor shape: " << src.sizes() 
               << " | Tensor strides: " << src.strides();
            throw std::runtime_error(ss.str());
        }

        h_jobs[i].src = src.data_ptr();
        h_jobs[i].dst = store_base_ptr + offsets[i];
        h_jobs[i].indices = flat_indices_ptr + indices_offsets[i];
        h_jobs[i].num_selected_pages = num_sel;
        h_jobs[i].is_page_first = is_pf ? 1 : 0;
        h_jobs[i].num_pages = num_pages;
        h_jobs[i].prefix_size = prefix_size;
        h_jobs[i].page_stride = page_stride;
        h_jobs[i].suffix_size = suffix_size;
        h_jobs[i].prefix_stride = prefix_stride;
        h_jobs[i].num_elements = num_elem;

        max_elements = std::max(max_elements, num_elem);
    }

    if (max_elements == 0) return;

    // Use the 16-bytes/thread int4 kernel when every job (and its store offset)
    // is 16-byte aligned; otherwise fall back to the 1-byte/thread kernel.
    bool aligned = true;
    for (size_t i = 0; i < num_jobs && aligned; ++i) {
        if (h_jobs[i].suffix_size % 16 != 0 || h_jobs[i].page_stride % 16 != 0 ||
            h_jobs[i].prefix_stride % 16 != 0 || h_jobs[i].num_elements % 16 != 0 ||
            offsets[i] % 16 != 0) {
            aligned = false;
        }
    }

    const int device_index = srcs[0].get_device();
    const cudaDeviceProp prop = get_device_properties(device_index);
    const int threads = 256;
    const long long unit = aligned ? 16 : 1;
    const long long blocks_x_i64 = (max_elements / unit + threads - 1) / threads;
    const int blocks_x = static_cast<int>(blocks_x_i64);
    const size_t max_jobs_per_launch = static_cast<size_t>(prop.maxGridSize[1]);

    auto d_jobs = torch::empty({static_cast<int64_t>(num_jobs * sizeof(PageCopyJobDesc))},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kCUDA, device_index)));

    record_tensor_on_external_stream(d_jobs, stream);
    CUDA_CHECK(cudaMemcpyAsync(d_jobs.data_ptr(), h_jobs.data(), num_jobs * sizeof(PageCopyJobDesc), cudaMemcpyHostToDevice, stream));

    const auto* jobs_ptr = reinterpret_cast<const PageCopyJobDesc*>(d_jobs.data_ptr());

    for (size_t job_begin = 0; job_begin < num_jobs; job_begin += max_jobs_per_launch) {
        const size_t jobs_this_launch = std::min(max_jobs_per_launch, num_jobs - job_begin);
        const dim3 grid(static_cast<unsigned int>(blocks_x), static_cast<unsigned int>(jobs_this_launch), 1);
        if (aligned) {
            fused_gather_pages_kernel_int4<<<grid, threads, 0, stream>>>(jobs_ptr + job_begin, static_cast<int>(jobs_this_launch));
        } else {
            fused_gather_pages_kernel<<<grid, threads, 0, stream>>>(jobs_ptr + job_begin, static_cast<int>(jobs_this_launch));
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

static void fused_scatter_pages_launch(
    const std::vector<torch::Tensor>& dsts, uint8_t* store_base_ptr,
    const std::vector<int64_t>& offsets, const std::vector<int64_t>& byte_sizes,
    const int* flat_indices_ptr, const std::vector<int64_t>& indices_offsets,
    const std::vector<char>& is_page_first_list, cudaStream_t stream
) {
    const size_t num_jobs = dsts.size();
    if (num_jobs == 0) return;

    std::vector<PageCopyJobDesc> h_jobs(num_jobs);
    long long max_elements = 0;

    for (size_t i = 0; i < num_jobs; ++i) {
        const auto& dst = dsts[i];
        const bool is_pf = static_cast<bool>(is_page_first_list[i]);

        int num_pages = 0, prefix_size = 0;
        long long page_stride = 0, suffix_size = 0, prefix_stride = 0;
        get_page_copy_shape(dst, is_pf, num_pages, prefix_size, page_stride, suffix_size, prefix_stride);

        const int64_t num_sel_i64 = indices_offsets[i + 1] - indices_offsets[i];
        const int num_sel = static_cast<int>(num_sel_i64);

        const long long num_elem = static_cast<long long>(prefix_size) * num_sel * suffix_size;

        if (static_cast<long long>(byte_sizes[i]) != num_elem) {
            std::stringstream ss;
            ss << "byte_size mismatch at job " << i 
               << " | C++ expected bytes: " << num_elem 
               << " | Python passed bytes: " << byte_sizes[i]
               << " | Tensor shape: " << dst.sizes() 
               << " | Tensor strides: " << dst.strides();
            throw std::runtime_error(ss.str());
        }

        h_jobs[i].src = store_base_ptr + offsets[i];
        h_jobs[i].dst = dst.data_ptr();
        h_jobs[i].indices = flat_indices_ptr + indices_offsets[i];
        h_jobs[i].num_selected_pages = num_sel;
        h_jobs[i].is_page_first = is_pf ? 1 : 0;
        h_jobs[i].num_pages = num_pages;
        h_jobs[i].prefix_size = prefix_size;
        h_jobs[i].page_stride = page_stride;
        h_jobs[i].suffix_size = suffix_size;
        h_jobs[i].prefix_stride = prefix_stride;
        h_jobs[i].num_elements = num_elem;

        max_elements = std::max(max_elements, num_elem);
    }

    if (max_elements == 0) return;

    const int device_index = dsts[0].get_device();
    const cudaDeviceProp prop = get_device_properties(device_index);
    const int threads = 256;
    const long long blocks_x_i64 = (max_elements + threads - 1) / threads;
    const int blocks_x = static_cast<int>(blocks_x_i64);
    const size_t max_jobs_per_launch = static_cast<size_t>(prop.maxGridSize[1]);

    auto d_jobs = torch::empty({static_cast<int64_t>(num_jobs * sizeof(PageCopyJobDesc))},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kCUDA, device_index)));

    record_tensor_on_external_stream(d_jobs, stream);
    CUDA_CHECK(cudaMemcpyAsync(d_jobs.data_ptr(), h_jobs.data(), num_jobs * sizeof(PageCopyJobDesc), cudaMemcpyHostToDevice, stream));

    const auto* jobs_ptr = reinterpret_cast<const PageCopyJobDesc*>(d_jobs.data_ptr());

    for (size_t job_begin = 0; job_begin < num_jobs; job_begin += max_jobs_per_launch) {
        const size_t jobs_this_launch = std::min(max_jobs_per_launch, num_jobs - job_begin);
        const dim3 grid(static_cast<unsigned int>(blocks_x), static_cast<unsigned int>(jobs_this_launch), 1);
        fused_scatter_pages_kernel<<<grid, threads, 0, stream>>>(jobs_ptr + job_begin, static_cast<int>(jobs_this_launch));
        CUDA_CHECK(cudaGetLastError());
    }
}

// ==================== DISPATCHERS ====================

void fused_gather_pages_dispatch(
    const std::vector<torch::Tensor>& srcs, uint8_t* store_base_ptr,
    const std::vector<int64_t>& offsets, const std::vector<int64_t>& byte_sizes,
    const int* flat_indices_ptr, const std::vector<int64_t>& indices_offsets,
    const std::vector<char>& is_page_first_list, cudaStream_t stream
) {
    TORCH_CHECK(!srcs.empty(), "no gather jobs");
    const auto device = srcs[0].device();

    for (size_t i = 0; i < srcs.size(); ++i) {
        TORCH_CHECK(srcs[i].is_cuda(), "all source layers must be CUDA tensors; job ", i);
        TORCH_CHECK(srcs[i].device() == device, "all source layers must be on the same device; job ", i);
    }

    c10::cuda::CUDAGuard device_guard(device);
    fused_gather_pages_launch(srcs, store_base_ptr, offsets, byte_sizes, flat_indices_ptr, indices_offsets, is_page_first_list, stream);
}

void fused_scatter_pages_dispatch(
    const std::vector<torch::Tensor>& dsts, uint8_t* store_base_ptr,
    const std::vector<int64_t>& offsets, const std::vector<int64_t>& byte_sizes,
    const int* flat_indices_ptr, const std::vector<int64_t>& indices_offsets,
    const std::vector<char>& is_page_first_list, cudaStream_t stream
) {
    TORCH_CHECK(!dsts.empty(), "no scatter jobs");
    const auto device = dsts[0].device();

    for (size_t i = 0; i < dsts.size(); ++i) {
        TORCH_CHECK(dsts[i].is_cuda(), "all destination layers must be CUDA tensors; job ", i);
        TORCH_CHECK(dsts[i].device() == device, "all destination layers must be on the same device; job ", i);
    }

    c10::cuda::CUDAGuard device_guard(device);
    fused_scatter_pages_launch(dsts, store_base_ptr, offsets, byte_sizes, flat_indices_ptr, indices_offsets, is_page_first_list, stream);
}