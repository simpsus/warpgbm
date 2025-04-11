from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

with open("version.txt", "r") as f:
    version = f.read().strip()

setup(
    name='warpgbm',
    version=version,
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='warpgbm.cuda.node_kernel',  # Matches import: from warpgbm.cuda import node_kernel
            sources=[
                'warpgbm/cuda/node_kernel.cpp',
                'warpgbm/cuda/histogram_kernel.cu',
                'warpgbm/cuda/best_split_kernel.cu'
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        'torch',
        'numpy',
        'scikit-learn',
        'tqdm',
    ],
    zip_safe=False,
)
