import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

def get_extensions():
    extensions = []

    if CUDA_HOME is not None:
        extensions.append(
            CUDAExtension(
                name="warpgbm.cuda.node_kernel",
                sources=[
                    "warpgbm/cuda/histogram_kernel.cu",
                    "warpgbm/cuda/best_split_kernel.cu",
                    "warpgbm/cuda/node_kernel.cpp",
                ]
            )
        )
    else:
        print("CUDA_HOME not found. Skipping CUDA extensions.")

    return extensions

# Get version
with open("version.txt") as f:
    version = f.read().strip()

setup(
    name="warpgbm",
    version=version,
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension} if CUDA_HOME is not None else {},
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn",
        "tqdm",
    ],
)
