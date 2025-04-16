import numpy as np
from warpgbm import WarpGBM
from sklearn.datasets import make_regression

def test_fit_predict_correlation():
    np.random.seed(42)
    N = 1_000_000
    F = 100
    X, y = make_regression(n_samples=N, n_features=F, noise=0.1, random_state=42)
    era = np.zeros(N, dtype=np.int32)
    corrs = []

    model = WarpGBM(
        max_depth = 10,
        num_bins = 10,
        n_estimators = 10,
        learning_rate = 1,
        verbosity=False,
        histogram_computer='hist1',
        threads_per_block=32,
        rows_per_thread=4
    )

    model.fit(X, y, era_id=era)
    preds = model.predict(X)

    # Pearson correlation in-sample
    corr = np.corrcoef(preds, y)[0, 1]
    corrs.append(corr)

    model = WarpGBM(
        max_depth = 10,
        num_bins = 10,
        n_estimators = 10,
        learning_rate = 1,
        verbosity=False,
        histogram_computer='hist2',
        threads_per_block=32,
        rows_per_thread=4
    )

    model.fit(X, y, era_id=era)
    preds = model.predict(X)

    # Pearson correlation in-sample
    corr = np.corrcoef(preds, y)[0, 1]
    corrs.append(corr)

    model = WarpGBM(
        max_depth = 10,
        num_bins = 10,
        n_estimators = 10,
        learning_rate = 1,
        verbosity=False,
        histogram_computer='hist3',
        threads_per_block=32,
        rows_per_thread=4
    )

    model.fit(X, y, era_id=era)
    preds = model.predict(X)

    # Pearson correlation in-sample
    corr = np.corrcoef(preds, y)[0, 1]
    corrs.append(corr)
    assert ( np.array(corrs) > 0.95 ).all(), f"In-sample correlation too low: {corr:.4f}"
