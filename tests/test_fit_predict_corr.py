import numpy as np
from warpgbm import WarpGBM
from sklearn.datasets import make_regression
import time
from sklearn.metrics import mean_squared_error


def test_fit_predictpytee_correlation():
    np.random.seed(42)
    N = 100_000
    F = 1000
    X, y = make_regression(n_samples=N, n_features=F, noise=0.1, random_state=42)
    era = np.zeros(N, dtype=np.int32)
    corrs = []
    mses = []

    for hist_type in ["hist1", "hist2", "hist3"]:
        print(f"\nTesting histogram method: {hist_type}")

        model = WarpGBM(
            max_depth=10,
            num_bins=10,
            n_estimators=100,
            learning_rate=1,
            verbosity=False,
            histogram_computer=hist_type,
            threads_per_block=64,
            rows_per_thread=4,
        )

        start_fit = time.time()
        model.fit(X, y, era_id=era)
        fit_time = time.time() - start_fit
        print(f"  Fit time:     {fit_time:.3f} seconds")

        start_pred = time.time()
        preds = model.predict(X)
        pred_time = time.time() - start_pred
        print(f"  Predict time: {pred_time:.3f} seconds")

        corr = np.corrcoef(preds, y)[0, 1]
        mse = mean_squared_error(preds, y)
        print(f"  Correlation:  {corr:.4f}")
        print(f"  MSE:  {mse:.4f}")
        corrs.append(corr)
        mses.append(mse)

    assert (np.array(corrs) > 0.9).all(), f"In-sample correlation too low: {corrs}"
    assert (np.array(mses) < 2).all(), f"In-sample mse too high: {mses}"
