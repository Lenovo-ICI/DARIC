#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <string>
#include <unordered_map>
#include <vector>
#include <limits>
#include <sstream>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t error = (call);                                             \
        if (error != cudaSuccess) {                                             \
            std::stringstream ss;                                               \
            ss << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - "   \
               << cudaGetErrorString(error)                                     \
               << " (error code " << static_cast<int>(error) << ")";           \
            throw std::runtime_error(ss.str());                                 \
        }                                                                       \
    } while (0)

struct PageCopyJobDesc {
    const void* src;
    void*       dst;
    const int*  indices;
    int         num_selected_pages;
    int         is_page_first;
    int         num_pages;
    int         prefix_size;
    long long   page_stride;
    long long   suffix_size;
    long long   num_elements;
    long long   prefix_stride;
};

// --- Helper Functions Declarations ---
void get_page_copy_shape(
    const torch::Tensor& tensor, bool is_page_first,
    int& num_pages, int& prefix_size, long long& page_stride,
    long long& suffix_size, long long& prefix_stride);

void validate_common_job_inputs(
    size_t num_jobs, const std::vector<int64_t>& offsets,
    const std::vector<int64_t>& byte_sizes,
    const std::vector<int64_t>& indices_offsets,
    const torch::Tensor& indices_flat);

cudaDeviceProp get_device_properties(int device_index);

void record_tensor_on_external_stream(
    const torch::Tensor& tensor, cudaStream_t stream);

// Batched copy-engine D2H: one cudaMemcpyAsync per (staging_offset, shm_offset,
// size) region on `stream_ptr` (0 = current stream). shm must be page-locked.
void batched_memcpy_d2h(
    torch::Tensor staging,
    torch::Tensor shm,
    const std::vector<int64_t>& stg_offsets,
    const std::vector<int64_t>& shm_offsets,
    const std::vector<int64_t>& sizes,
    uintptr_t stream_ptr);

// Batched copy-engine H2D: one cudaMemcpyAsync per (shm_offset, staging_offset,
// size) region on `stream_ptr` (0 = current stream). shm must be page-locked.
// Used by the staged read path on platforms where SM zero-copy reads of
// registered host memory fault (e.g. Hygon DCU / DTK).
void batched_memcpy_h2d(
    torch::Tensor shm,
    torch::Tensor staging,
    const std::vector<int64_t>& shm_offsets,
    const std::vector<int64_t>& stg_offsets,
    const std::vector<int64_t>& sizes,
    uintptr_t stream_ptr);

// --- Dispatchers Declarations ---
void fused_gather_pages_dispatch(
    const std::vector<torch::Tensor>& srcs, uint8_t* store_base_ptr,
    const std::vector<int64_t>& offsets, const std::vector<int64_t>& byte_sizes,
    const int* flat_indices_ptr, const std::vector<int64_t>& indices_offsets,
    const std::vector<char>& is_page_first_list, cudaStream_t stream);

void fused_scatter_pages_dispatch(
    const std::vector<torch::Tensor>& dsts, uint8_t* store_base_ptr,
    const std::vector<int64_t>& offsets, const std::vector<int64_t>& byte_sizes,
    const int* flat_indices_ptr, const std::vector<int64_t>& indices_offsets,
    const std::vector<char>& is_page_first_list, cudaStream_t stream);

// --- CacheCopier Class ---
class CacheCopier {
public:
    void register_layer(const std::string& name, torch::Tensor tensor, bool is_page_first);
    void register_store(torch::Tensor buffer);
    void gather_pages(
        const std::vector<std::string>& layer_names,
        const std::vector<int64_t>& offsets,
        const std::vector<int64_t>& byte_sizes,
        const torch::Tensor& indices_flat,
        const std::vector<int64_t>& indices_offsets,
        uintptr_t stream_ptr);
    void scatter_pages(
        const std::vector<std::string>& layer_names,
        const std::vector<int64_t>& offsets,
        const std::vector<int64_t>& byte_sizes,
        const torch::Tensor& indices_flat,
        const std::vector<int64_t>& indices_offsets,
        uintptr_t stream_ptr);

private:
    struct LayerInfo {
        torch::Tensor tensor;
        char is_page_first;
    };

    cudaStream_t resolve_stream(uintptr_t stream_ptr, int device_index) const;
    void validate_store_ranges(
        const std::vector<int64_t>& offsets,
        const std::vector<int64_t>& byte_sizes) const;

    std::unordered_map<std::string, LayerInfo> layers_;
    torch::Tensor store_buffer_;
    uint8_t* store_base_ptr_ = nullptr;
    int64_t store_size_bytes_ = 0;
    bool has_store_ = false;
};