import numpy as np
from warpgbm import WarpGBM

def test_fit_predict_correlation():
    np.random.seed(42)

    N = 200
    F = 5
    X = np.random.randn(N, F).astype(np.float32)
    true_weights = np.array([0.5, -1.0, 2.0, 0.0, 1.0])
    noise = 0.1 * np.random.randn(N)
    y = (X @ true_weights + noise).astype(np.float32)
    era = np.zeros(N, dtype=np.int32)

    model = WarpGBM(
        num_bins=16,
        max_depth=3,
        n_estimators=10,
        learning_rate=0.2,
        verbosity=False,
        device='cuda'
    )

    model.fit(X, y, era_id=era)
    preds = model.predict(X)

    # Pearson correlation in-sample
    corr = np.corrcoef(preds, y)[0, 1]
    assert corr > 0.95, f"In-sample correlation too low: {corr:.4f}"
