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

### 🔧 Recommended (GitHub, always latest):

```bash
pip install git+https://github.com/jefferythewind/warpgbm.git
```

This installs the latest version directly from GitHub and compiles CUDA extensions on your machine using your **local PyTorch and CUDA setup**. It's the most reliable method for ensuring compatibility and staying up to date with the latest features.

### 📦 Alternatively (PyPI, stable releases):

```bash
pip install warpgbm
```

This installs from PyPI and also compiles CUDA code locally during installation. This method works well **if your environment already has PyTorch with GPU support** installed and configured.

> 💡 **Tip:**\
> If you encounter an error related to mismatched or missing CUDA versions, try installing with the following flag:
>
> ```bash
> pip install warpgbm --no-build-isolation
> ```

Before either method, make sure you’ve installed PyTorch with GPU support:\
👉 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## Example

```python
import numpy as np
from sklearn.datasets import make_regression
from time import time
import lightgbm as lgb
from warpgbm import WarpGBM

# Create synthetic regression dataset
X, y = make_regression(n_samples=100_000, n_features=500, noise=0.1, random_state=42)
X = X.astype(np.float32)
y = y.astype(np.float32)

# Train LightGBM
start = time()
lgb_model = lgb.LGBMRegressor(max_depth=5, n_estimators=100, learning_rate=0.01, max_bin=7)
lgb_model.fit(X, y)
lgb_time = time() - start
lgb_preds = lgb_model.predict(X)

# Train WarpGBM
start = time()
wgbm_model = WarpGBM(max_depth=5, n_estimators=100, learning_rate=0.01, num_bins=7)
wgbm_model.fit(X, y)
wgbm_time = time() - start
wgbm_preds = wgbm_model.predict(X)

# Results
print(f"LightGBM:   corr = {np.corrcoef(lgb_preds, y)[0,1]:.4f}, time = {lgb_time:.2f}s")
print(f"WarpGBM:     corr = {np.corrcoef(wgbm_preds, y)[0,1]:.4f}, time = {wgbm_time:.2f}s")
```

**🧪 Results (Ryzen 9 CPU, NVIDIA 3090 GPU):**

```
LightGBM:   corr = 0.8742, time = 37.33s
WarpGBM:     corr = 0.8621, time = 5.40s
```

---

## Pre-binned Data Example (Numerai)

WarpGBM can save additional training time if your dataset is already pre-binned. The Numerai tournament data is a great example:

```python
import pandas as pd
from numerapi import NumerAPI
from time import time
import lightgbm as lgb
from warpgbm import WarpGBM
import numpy as np

napi = NumerAPI()
napi.download_dataset('v5.0/train.parquet', 'train.parquet')
train = pd.read_parquet('train.parquet')

feature_set = [f for f in train.columns if 'feature' in f]
target = 'target_cyrus'

X_np = train[feature_set].astype('int8').values
Y_np = train[target].values

# LightGBM
start = time()
lgb_model = lgb.LGBMRegressor(max_depth=5, n_estimators=100, learning_rate=0.01, max_bin=7)
lgb_model.fit(X_np, Y_np)
lgb_time = time() - start
lgb_preds = lgb_model.predict(X_np)

# WarpGBM
start = time()
wgbm_model = WarpGBM(max_depth=5, n_estimators=100, learning_rate=0.01, num_bins=7)
wgbm_model.fit(X_np, Y_np)
wgbm_time = time() - start
wgbm_preds = wgbm_model.predict(X_np)

# Results
print(f"LightGBM:   corr = {np.corrcoef(lgb_preds, Y_np)[0,1]:.4f}, time = {lgb_time:.2f}s")
print(f"WarpGBM:     corr = {np.corrcoef(wgbm_preds, Y_np)[0,1]:.4f}, time = {wgbm_time:.2f}s")
```

---

## Documentation

### `WarpGBM` Parameters:
- `num_bins`: Number of histogram bins to use (default: 10)
- `max_depth`: Maximum depth of trees (default: 3)
- `learning_rate`: Shrinkage rate applied to leaf outputs (default: 0.1)
- `n_estimators`: Number of boosting iterations (default: 100)
- `min_child_weight`: Minimum sum of instance weight needed in a child (default: 20)
- `min_split_gain`: Minimum loss reduction required to make a further partition (default: 0.0)
- `verbosity`: Whether to print training logs (default: True)
- `histogram_computer`: Choice of histogram kernel (`'hist1'`, `'hist2'`, `'hist3'`) (default: `'hist3'`)
- `threads_per_block`: CUDA threads per block (default: 32)
- `rows_per_thread`: Number of training rows processed per thread (default: 4)
- `device`: Device to train on (`'cuda'` or `'cpu'`, default: `'cuda'`)
- `split_type`: Algorithm used to choose best split (`'v1'` = CUDA kernel, `'v2'` = torch-based) (default: `'v2'`)

### Methods:
- `.fit(X, y, era_id=None)`: Train the model. `X` can be raw floats or pre-binned `int8` data. `era_id` is optional and used internally.
- `.predict(X)`: Predict on new raw float or pre-binned data.
- `.predict_data(bin_indices)`: Predict from binned data directly (NumPy `int8` matrix).
- `.grow_forest()`: Manually triggers tree construction loop (usually not needed).

---

## Acknowledgements

WarpGBM builds on the shoulders of PyTorch, scikit-learn, LightGBM, and the CUDA ecosystem. Thanks to all contributors in the GBDT research and engineering space.

---

