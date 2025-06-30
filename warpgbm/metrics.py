# warpgbm/metrics.py

import torch

def rmsle_torch(y_true, y_pred, eps=1e-7):
    y_true = torch.clamp(y_true, min=0)
    y_pred = torch.clamp(y_pred, min=0)
    log_true = torch.log1p(y_true + eps)
    log_pred = torch.log1p(y_pred + eps)
    return torch.sqrt(torch.mean((log_true - log_pred) ** 2))

def get_eval_metric(metric, y_true, y_pred):
    if metric == "mse":
        return ((y_true - y_pred) ** 2).mean().item()
    elif metric == "corr":
        return 1 - torch.corrcoef(torch.vstack([y_true, y_pred]))[0, 1].item()
    elif metric == "rmsle":
        return rmsle_torch(y_true, y_pred).item()
    else:
        raise ValueError(f"Invalid eval_metric: {metric}.")