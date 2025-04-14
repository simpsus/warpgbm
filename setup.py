import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

# Check CUDA availability
if not torch.cuda.is_available():
    print("CUDA is not available. Please ensure you have a compatible GPU and CUDA setup.")
    sys.exit(1)

# Get the CUDA version from PyTorch
cuda_version = torch.version.cuda
print(f"Building with CUDA version: {cuda_version}")

def get_extensions():
    return [
        CUDAExtension(
            name="warpgbm.cuda.node_kernel",
            sources=[
                "warpgbm/cuda/histogram_kernel.cu",
                "warpgbm/cuda/best_split_kernel.cu",
                "warpgbm/cuda/node_kernel.cpp",
            ]
        )
    ]

with open("version.txt") as f:
    version = f.read().strip()

setup(
    name="warpgbm",
    version=version,
    packages=find_packages(),  #auto-includes warpgbm.cuda
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn",
        "tqdm"
    ],
)
