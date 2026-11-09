#include "cache_ops.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>
#include <iostream>

void CacheCopier::register_layer(const std::string& name, torch::Tensor tensor, bool is_page_first) {
    TORCH_CHECK(tensor.is_cuda(), "kvcache tensor must be on GPU: ", name);
              
    layers_.insert_or_assign(name, LayerInfo{std::move(tensor), static_cast<char>(is_page_first ? 1 : 0)});
}

void CacheCopier::register_store(torch::Tensor buffer) {
    // The store buffer is GPU staging memory; separate copy-engine DMA handles
    // transfers between staging and host shared memory.
    TORCH_CHECK(buffer.is_cuda(), "staging buffer must be on CUDA");
    TORCH_CHECK(buffer.dim() == 1, "store buffer must be 1D");
    TORCH_CHECK(buffer.dtype() == torch::kUInt8, "store buffer must be uint8");
    TORCH_CHECK(buffer.is_contiguous(), "store buffer must be contiguous");

    store_buffer_ = std::move(buffer);
    store_base_ptr_ = reinterpret_cast<uint8_t*>(store_buffer_.data_ptr());
    store_size_bytes_ = store_buffer_.numel();
    has_store_ = true;
}

cudaStream_t CacheCopier::resolve_stream(uintptr_t stream_ptr, int device_index) const {
    if (stream_ptr != 0) return reinterpret_cast<cudaStream_t>(stream_ptr);
    return at::cuda::getCurrentCUDAStream(device_index).stream();
}

void CacheCopier::validate_store_ranges(const std::vector<int64_t>& offsets, const std::vector<int64_t>& byte_sizes) const {
    for (size_t i = 0; i < offsets.size(); ++i) {
        TORCH_CHECK(offsets[i] >= 0, "negative store offset at job ", i);
        TORCH_CHECK(byte_sizes[i] >= 0, "negative store byte size at job ", i);
        TORCH_CHECK(offsets[i] <= store_size_bytes_, "store offset out of range at job ", i);
        TORCH_CHECK(byte_sizes[i] <= store_size_bytes_ - offsets[i], "store range out of bounds at job ", i);
    }
}

void CacheCopier::gather_pages(
    const std::vector<std::string>& layer_names, const std::vector<int64_t>& offsets,
    const std::vector<int64_t>& byte_sizes, const torch::Tensor& indices_flat,
    const std::vector<int64_t>& indices_offsets, uintptr_t stream_ptr
) {
    TORCH_CHECK(has_store_, "store buffer not registered; call register_store() first");
    const size_t num_jobs = layer_names.size();
    if (num_jobs == 0) return;

    validate_common_job_inputs(num_jobs, offsets, byte_sizes, indices_offsets, indices_flat);
    validate_store_ranges(offsets, byte_sizes);

    std::vector<torch::Tensor> srcs;
    std::vector<char> is_pf;
    srcs.reserve(num_jobs);
    is_pf.reserve(num_jobs);

    for (const auto& name : layer_names) {
        const auto it = layers_.find(name);
        TORCH_CHECK(it != layers_.end(), "layer not registered: ", name);
        srcs.push_back(it->second.tensor);
        is_pf.push_back(it->second.is_page_first);
    }

    TORCH_CHECK(indices_flat.device() == srcs[0].device(), "indices_flat must be on the same device as kvcache");
    c10::cuda::CUDAGuard device_guard(srcs[0].device());
    cudaStream_t stream = resolve_stream(stream_ptr, srcs[0].get_device());

    record_tensor_on_external_stream(indices_flat, stream);
    {
        // Release the GIL during the kernel launch so an in-process async
        // writer thread's GPU work doesn't block the forward thread's Python.
        py::gil_scoped_release release;
        fused_gather_pages_dispatch(srcs, store_base_ptr_, offsets, byte_sizes, indices_flat.data_ptr<int>(), indices_offsets, is_pf, stream);
    }
}

void CacheCopier::scatter_pages(
    const std::vector<std::string>& layer_names, const std::vector<int64_t>& offsets,
    const std::vector<int64_t>& byte_sizes, const torch::Tensor& indices_flat,
    const std::vector<int64_t>& indices_offsets, uintptr_t stream_ptr
) {
    TORCH_CHECK(has_store_, "store buffer not registered; call register_store() first");
    const size_t num_jobs = layer_names.size();
    if (num_jobs == 0) return;

    validate_common_job_inputs(num_jobs, offsets, byte_sizes, indices_offsets, indices_flat);
    validate_store_ranges(offsets, byte_sizes);

    std::vector<torch::Tensor> dsts;
    std::vector<char> is_pf;
    dsts.reserve(num_jobs);
    is_pf.reserve(num_jobs);

    for (const auto& name : layer_names) {
        const auto it = layers_.find(name);
        TORCH_CHECK(it != layers_.end(), "layer not registered: ", name);
        dsts.push_back(it->second.tensor);
        is_pf.push_back(it->second.is_page_first);
    }

    TORCH_CHECK(indices_flat.device() == dsts[0].device(), "indices_flat must be on the same device as kvcache");
    c10::cuda::CUDAGuard device_guard(dsts[0].device());
    cudaStream_t stream = resolve_stream(stream_ptr, dsts[0].get_device());

    record_tensor_on_external_stream(indices_flat, stream);
    {
        py::gil_scoped_release release;
        fused_scatter_pages_dispatch(dsts, store_base_ptr_, offsets, byte_sizes, indices_flat.data_ptr<int>(), indices_offsets, is_pf, stream);
    }
}