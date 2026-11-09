#include "cache_ops.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(cache_ops, m) {
    m.doc() = "Staged KV-cache gather/scatter and batched host/device copy operations";

    py::class_<CacheCopier>(m, "CacheCopier")
        .def(py::init<>())
        .def("register_layer", &CacheCopier::register_layer, py::arg("name"), py::arg("tensor"), py::arg("is_page_first"))
        .def("register_store", &CacheCopier::register_store, py::arg("buffer"))
        .def("gather_pages", &CacheCopier::gather_pages, py::arg("layer_names"), py::arg("offsets"), py::arg("byte_sizes"), py::arg("indices_flat"), py::arg("indices_offsets"), py::arg("stream") = 0)
        .def("scatter_pages", &CacheCopier::scatter_pages, py::arg("layer_names"), py::arg("offsets"), py::arg("byte_sizes"), py::arg("indices_flat"), py::arg("indices_offsets"), py::arg("stream") = 0);

    m.def("batched_memcpy_d2h", &batched_memcpy_d2h,
          py::arg("staging"), py::arg("shm"),
          py::arg("stg_offsets"), py::arg("shm_offsets"), py::arg("sizes"),
          py::arg("stream") = 0,
          "Batched copy-engine cudaMemcpyAsync (D2H) per region. shm must be "
          "page-locked (cudaHostRegister). Bypasses PyTorch is_pinned().");

    m.def("batched_memcpy_h2d", &batched_memcpy_h2d,
          py::arg("shm"), py::arg("staging"),
          py::arg("shm_offsets"), py::arg("stg_offsets"), py::arg("sizes"),
          py::arg("stream") = 0,
          "Batched copy-engine cudaMemcpyAsync (H2D) per region. shm must be "
          "page-locked (cudaHostRegister). Bypasses PyTorch is_pinned().");

}