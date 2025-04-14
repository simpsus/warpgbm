import numpy as np
from warpgbm import WarpGBM

def test_fit_predict_correlation():
    np.random.seed(42)
    N = 500
    F = 5
    X = np.random.randn(N, F).astype(np.float32)
    true_weights = np.array([0.5, -1.0, 2.0, 0.0, 1.0])
    noise = 0.1 * np.random.randn(N)
    y = (X @ true_weights + noise).astype(np.float32)
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
