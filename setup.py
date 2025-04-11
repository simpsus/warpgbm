import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
    packages=["warpgbm"],
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn",
        "tqdm"
    ],
)
