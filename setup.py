from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


DEPENDENCIES = ["xxhash", "msgpack", "pandas", "redis", "httpx", "torch"]


cuda_ext = CUDAExtension(
    name="daric.cache_ops",
    sources=[
        "csrc/cache_ops/bindings.cpp",
        "csrc/cache_ops/copier.cpp",
        "csrc/cache_ops/helpers.cu",
        "csrc/cache_ops/kernels.cu",
    ],
    include_dirs=["csrc/cache_ops"],
    extra_compile_args={
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
        ],
    },
)


setup(
    name="daric",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[cuda_ext],
    cmdclass={"build_ext": BuildExtension},
    install_requires=DEPENDENCIES,
    zip_safe=False,
)