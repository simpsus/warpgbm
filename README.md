# WarpGBM

WarpGBM is a high-performance, GPU-accelerated Gradient Boosted Decision Tree (GBDT) library built with PyTorch and CUDA. It offers blazing-fast histogram-based training and efficient prediction, with compatibility for research and production workflows.

---

## Features

- GPU-accelerated training and histogram construction using custom CUDA kernels
- Drop-in scikit-learn style interface
- Supports pre-binned data or automatic quantile binning
- Fully differentiable prediction path
- Simple install with `pip`

---

## Performance Note

In our initial tests on an NVIDIA 3090 (local) and A100 (Google Colab Pro), WarpGBM achieves **14x to 20x faster training times** compared to LightGBM using default configurations. It also consumes **significantly less RAM and CPU**. These early results hint at more thorough benchmarking to come.

---

## Installation

First, install PyTorch for your system with GPU support:  
https://pytorch.org/get-started/locally/

Then:

```bash
pip install warpgbm
```

Note: WarpGBM will compile custom CUDA extensions at install time using your installed PyTorch.

---

## Example

```python
import numpy as np
from warpgbm import WarpGBM

# Generate a simple regression dataset
X = np.random.randn(100, 5).astype(np.float32)
w = np.array([0.5, -1.0, 2.0, 0.0, 1.0])
y = (X @ w + 0.1 * np.random.randn(100)).astype(np.float32)

model = WarpGBM(max_depth=3, n_estimators=10)
model.fit(X, y)  # era_id is optional
preds = model.predict(X)
```

---

## Acknowledgements

WarpGBM builds on the shoulders of PyTorch, scikit-learn, LightGBM, and the CUDA ecosystem. Thanks to all contributors in the GBDT research and engineering space.

---
