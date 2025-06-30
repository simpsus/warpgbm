import torch
import numpy as np
from warpgbm import WarpGBM
from warpgbm.metrics import get_eval_metric
from abc import ABC
from loguru import logger


class EarlyStoppingCallback(ABC):

    def evaluate_stopping(self, model:WarpGBM)->bool:
        pass

class NTreesMRoundsESCB(EarlyStoppingCallback):

    def __init__(self, X_eval, y_eval, eval_metric="mse", early_stopping_rounds=10, eval_every_n_trees=1, verbose=True):
        if not eval_metric in ['mse', 'rmsle', 'corr']:
            raise ValueError(f"unsupported eval_metric {eval_metric}. Must be one of ['mse', 'rmsle', 'corr']")
        if not eval_every_n_trees <= early_stopping_rounds:
            raise ValueError(f"eval_every_n_trees must be <= {early_stopping_rounds}.")
        if not isinstance(X_eval, np.ndarray) or not isinstance(y_eval, np.ndarray):
            raise TypeError("X_eval and y_eval must be numpy arrays.")
        if X_eval.ndim != 2:
            raise ValueError(
                f"X_eval must be 2-dimensional, got shape {X_eval.shape}"
            )
        if y_eval.ndim != 1:
            raise ValueError(
                f"y_eval must be 1-dimensional, got shape {y_eval.shape}"
            )
        if X_eval.shape[0] != y_eval.shape[0]:
            raise ValueError(
                f"X_eval and y_eval must have same number of rows. Got {X_eval.shape[0]} and {y_eval.shape[0]}."
            )
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_every_n_trees = eval_every_n_trees
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.Y_gpu_eval = None
        self.bin_indices_eval = None
        self.eval_loss = []
        self.eval_status = ""
        self.verbose = verbose

    def evaluate_stopping(self, model:WarpGBM)->bool:
        if self.Y_gpu_eval is None:
            self.Y_gpu_eval = torch.from_numpy(self.y_eval).to(torch.float32).to(model.device)
        if self.bin_indices_eval is None:
            self.bin_indices_eval = model.bin_inference_data(self.X_eval)
        number_trees = len(model.forest)
        stop = False
        if number_trees % self.eval_every_n_trees == 0:
            debug_msg = f"Evaluating Early Stopping as {number_trees}%{self.eval_every_n_trees} == 0. "
            eval_preds = model.predict_binned(self.bin_indices_eval)
            eval_loss = get_eval_metric(self.eval_metric, self.Y_gpu_eval, eval_preds )
            self.eval_loss.append(eval_loss)
            if len(self.eval_loss) > self.early_stopping_rounds:
                debug_msg += f"There are {len(self.eval_loss)} evaluation losses, so more than {self.early_stopping_rounds} early stopping rounds. "
                if self.eval_loss[-(self.early_stopping_rounds+1)] < self.eval_loss[-1]:
                    debug_msg += f"STOPPING as loss @{-(self.early_stopping_rounds+1)} ({self.eval_loss[-(self.early_stopping_rounds+1)]}) is less than the most recent one ({self.eval_loss[-1]})"
                    stop = True
            if self.verbose:
                logger.debug(debug_msg)
            self.eval_status = f"🌲 Tree {number_trees}/{model.n_estimators} | Train MSE: {model.training_loss[-1]:.6f} | Eval {self.eval_metric}: {eval_loss:.6f}"
            del eval_preds, eval_loss
        return stop


X = np.random.rand(1000,1000)
y = np.random.rand(1000)
X_eval = np.random.rand(1000,1000)
y_eval = np.random.rand(1000)

es = NTreesMRoundsESCB(X_eval, y_eval) # should terminate soon because completely random data
model = WarpGBM()
model.fit(X,y, es_callbacks =[es])
