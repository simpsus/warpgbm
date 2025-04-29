import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from warpgbm.cuda import node_kernel
from tqdm import tqdm
from typing import Tuple
from torch import Tensor

histogram_kernels = {
    "hist1": node_kernel.compute_histogram,
    "hist2": node_kernel.compute_histogram2,
    "hist3": node_kernel.compute_histogram3,
}


class WarpGBM(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        num_bins=10,
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        min_child_weight=20,
        min_split_gain=0.0,
        verbosity=True,
        histogram_computer="hist3",
        threads_per_block=64,
        rows_per_thread=4,
        L2_reg=1e-6,
        L1_reg=0.0,
        device="cuda",
    ):
        # Validate arguments
        self._validate_hyperparams(
            num_bins=num_bins,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            min_split_gain=min_split_gain,
            histogram_computer=histogram_computer,
            threads_per_block=threads_per_block,
            rows_per_thread=rows_per_thread,
            L2_reg=L2_reg,
            L1_reg=L1_reg,
        )

        self.num_bins = num_bins
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.forest = None
        self.bin_edges = None
        self.base_prediction = None
        self.unique_eras = None
        self.device = device
        self.root_gradient_histogram = None
        self.root_hessian_histogram = None
        self.gradients = None
        self.root_node_indices = None
        self.bin_indices = None
        self.Y_gpu = None
        self.num_features = None
        self.num_samples = None
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.best_bin = torch.tensor([-1], dtype=torch.int32, device=self.device)
        self.compute_histogram = histogram_kernels[histogram_computer]
        self.threads_per_block = threads_per_block
        self.rows_per_thread = rows_per_thread
        self.L2_reg = L2_reg
        self.L1_reg = L1_reg

    def _validate_hyperparams(self, **kwargs):
        # Type checks
        int_params = [
            "num_bins",
            "max_depth",
            "n_estimators",
            "min_child_weight",
            "threads_per_block",
            "rows_per_thread",
        ]
        float_params = ["learning_rate", "min_split_gain", "L2_reg", "L1_reg"]

        for param in int_params:
            if not isinstance(kwargs[param], int):
                raise TypeError(
                    f"{param} must be an integer, got {type(kwargs[param])}."
                )

        for param in float_params:
            if not isinstance(
                kwargs[param], (float, int)
            ):  # Accept ints as valid floats
                raise TypeError(f"{param} must be a float, got {type(kwargs[param])}.")

        if not (2 <= kwargs["num_bins"] <= 127):
            raise ValueError("num_bins must be between 2 and 127 inclusive.")
        if kwargs["max_depth"] < 1:
            raise ValueError("max_depth must be at least 1.")
        if not (0.0 < kwargs["learning_rate"] <= 1.0):
            raise ValueError("learning_rate must be in (0.0, 1.0].")
        if kwargs["n_estimators"] <= 0:
            raise ValueError("n_estimators must be positive.")
        if kwargs["min_child_weight"] < 1:
            raise ValueError("min_child_weight must be a positive integer.")
        if kwargs["min_split_gain"] < 0:
            raise ValueError("min_split_gain must be non-negative.")
        if kwargs["threads_per_block"] <= 0 or kwargs["threads_per_block"] % 32 != 0:
            raise ValueError(
                "threads_per_block should be a positive multiple of 32 (warp size)."
            )
        if not (1 <= kwargs["rows_per_thread"] <= 16):
            raise ValueError(
                "rows_per_thread must be positive between 1 and 16 inclusive."
            )
        if kwargs["L2_reg"] < 0 or kwargs["L1_reg"] < 0:
            raise ValueError("L2_reg and L1_reg must be non-negative.")
        if kwargs["histogram_computer"] not in histogram_kernels:
            raise ValueError(
                f"Invalid histogram_computer: {kwargs['histogram_computer']}. Choose from {list(histogram_kernels.keys())}."
            )

    def fit(self, X, y, era_id=None):
        if era_id is None:
            era_id = np.ones(X.shape[0], dtype="int32")
        self.bin_indices, era_indices, self.bin_edges, self.unique_eras, self.Y_gpu = (
            self.preprocess_gpu_data(X, y, era_id)
        )
        self.num_samples, self.num_features = X.shape
        self.gradients = torch.zeros_like(self.Y_gpu)
        self.root_node_indices = torch.arange(self.num_samples, device=self.device)
        self.base_prediction = self.Y_gpu.mean().item()
        self.gradients += self.base_prediction
        self.best_gains = torch.zeros(self.num_features, device=self.device)
        self.best_bins = torch.zeros(
            self.num_features, device=self.device, dtype=torch.int32
        )
        with torch.no_grad():
            self.forest = self.grow_forest()
        return self

    def preprocess_gpu_data(self, X_np, Y_np, era_id_np):
        with torch.no_grad():
            self.num_samples, self.num_features = X_np.shape
            Y_gpu = torch.from_numpy(Y_np).type(torch.float32).to(self.device)
            era_id_gpu = torch.from_numpy(era_id_np).type(torch.int32).to(self.device)
            is_integer_type = np.issubdtype(X_np.dtype, np.integer)
            if is_integer_type:
                max_vals = X_np.max(axis=0)
                if np.all(max_vals < self.num_bins):
                    print(
                        "Detected pre-binned integer input — skipping quantile binning."
                    )
                    bin_indices = (
                        torch.from_numpy(X_np)
                        .to(self.device)
                        .contiguous()
                        .to(torch.int8)
                    )

                    # We'll store None or an empty tensor in self.bin_edges
                    # to indicate that we skip binning at predict-time
                    bin_edges = torch.arange(
                        1, self.num_bins, dtype=torch.float32
                    ).repeat(self.num_features, 1)
                    bin_edges = bin_edges.to(self.device)
                    unique_eras, era_indices = torch.unique(
                        era_id_gpu, return_inverse=True
                    )
                    return bin_indices, era_indices, bin_edges, unique_eras, Y_gpu
                else:
                    print(
                        "Integer input detected, but values exceed num_bins — falling back to quantile binning."
                    )

            bin_indices = torch.empty(
                (self.num_samples, self.num_features), dtype=torch.int8, device="cuda"
            )
            bin_edges = torch.empty(
                (self.num_features, self.num_bins - 1),
                dtype=torch.float32,
                device="cuda",
            )

            X_np = torch.from_numpy(X_np).to(torch.float32).pin_memory()

            for f in range(self.num_features):
                X_f = X_np[:, f].to("cuda", non_blocking=True)
                quantiles = torch.linspace(
                    0, 1, self.num_bins + 1, device="cuda", dtype=X_f.dtype
                )[1:-1]
                bin_edges_f = torch.quantile(
                    X_f, quantiles, dim=0
                ).contiguous()  # shape: [B-1] for 1D input
                bin_indices_f = bin_indices[:, f].contiguous()  # view into output
                node_kernel.custom_cuda_binner(X_f, bin_edges_f, bin_indices_f)
                bin_indices[:, f] = bin_indices_f
                bin_edges[f, :] = bin_edges_f

            unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)
            return bin_indices, era_indices, bin_edges, unique_eras, Y_gpu

    def compute_histograms(self, bin_indices_sub, gradients):
        grad_hist = torch.zeros(
            (self.num_features, self.num_bins), device=self.device, dtype=torch.float32
        )
        hess_hist = torch.zeros(
            (self.num_features, self.num_bins), device=self.device, dtype=torch.float32
        )

        self.compute_histogram(
            bin_indices_sub,
            gradients,
            grad_hist,
            hess_hist,
            self.num_bins,
            self.threads_per_block,
            self.rows_per_thread,
        )
        return grad_hist, hess_hist

    def find_best_split(self, gradient_histogram, hessian_histogram):
        node_kernel.compute_split(
            gradient_histogram,
            hessian_histogram,
            self.min_split_gain,
            self.min_child_weight,
            self.L2_reg,
            self.best_gains,
            self.best_bins,
            self.threads_per_block,
        )

        if torch.all(self.best_bins == -1):
            return -1, -1  # No valid split found

        f = torch.argmax(self.best_gains).item()
        b = self.best_bins[f].item()

        return f, b

    def grow_tree(self, gradient_histogram, hessian_histogram, node_indices, depth):
        if depth == self.max_depth:
            leaf_value = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * leaf_value
            return {"leaf_value": leaf_value.item(), "samples": node_indices.numel()}

        parent_size = node_indices.numel()
        best_feature, best_bin = self.find_best_split(
            gradient_histogram, hessian_histogram
        )

        if best_feature == -1:
            leaf_value = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * leaf_value
            return {"leaf_value": leaf_value.item(), "samples": parent_size}

        split_mask = self.bin_indices[node_indices, best_feature] <= best_bin
        left_indices = node_indices[split_mask]
        right_indices = node_indices[~split_mask]

        left_size = left_indices.numel()
        right_size = right_indices.numel()

        if left_size <= right_size:
            grad_hist_left, hess_hist_left = self.compute_histograms(
                self.bin_indices[left_indices], self.residual[left_indices]
            )
            grad_hist_right = gradient_histogram - grad_hist_left
            hess_hist_right = hessian_histogram - hess_hist_left
        else:
            grad_hist_right, hess_hist_right = self.compute_histograms(
                self.bin_indices[right_indices], self.residual[right_indices]
            )
            grad_hist_left = gradient_histogram - grad_hist_right
            hess_hist_left = hessian_histogram - hess_hist_right

        new_depth = depth + 1
        left_child = self.grow_tree(
            grad_hist_left, hess_hist_left, left_indices, new_depth
        )
        right_child = self.grow_tree(
            grad_hist_right, hess_hist_right, right_indices, new_depth
        )

        return {
            "feature": best_feature,
            "bin": best_bin,
            "left": left_child,
            "right": right_child,
        }

    def grow_forest(self):
        forest = [{} for _ in range(self.n_estimators)]
        self.training_loss = []

        for i in tqdm(range(self.n_estimators)):
            self.residual = self.Y_gpu - self.gradients

            self.root_gradient_histogram, self.root_hessian_histogram = (
                self.compute_histograms(self.bin_indices, self.residual)
            )

            tree = self.grow_tree(
                self.root_gradient_histogram,
                self.root_hessian_histogram,
                self.root_node_indices,
                depth=0,
            )
            forest[i] = tree
        # loss = ((self.Y_gpu - self.gradients) ** 2).mean().item()
        # self.training_loss.append(loss)
        # print(f"🌲 Tree {i+1}/{self.n_estimators} - MSE: {loss:.6f}")

        print("Finished training forest.")
        return forest

    def predict(self, X_np):
        X_tensor = torch.from_numpy(X_np).to(torch.float32).pin_memory()
        num_samples = X_tensor.size(0)
        bin_indices = torch.zeros(
            (num_samples, self.num_features), dtype=torch.int8, device=self.device
        )

        with torch.no_grad():
            for f in range(self.num_features):
                X_f = X_tensor[:, f].to(self.device, non_blocking=True)
                bin_edges_f = self.bin_edges[f]
                bin_indices_f = bin_indices[:, f].contiguous()
                node_kernel.custom_cuda_binner(X_f, bin_edges_f, bin_indices_f)
                bin_indices[:, f] = bin_indices_f

        tree_tensor = torch.stack(
            [
                self.flatten_tree(tree, max_nodes=2 ** (self.max_depth + 1))
                for tree in self.forest
            ]
        ).to(self.device)

        out = torch.zeros(num_samples, device=self.device) + self.base_prediction
        node_kernel.predict_forest(
            bin_indices.contiguous(), tree_tensor.contiguous(), self.learning_rate, out
        )

        return out.cpu().numpy()

    def flatten_tree(self, tree, max_nodes):
        """
        Convert a recursive tree structure into a flat matrix format.

        Each row in the output represents a node:
        - Columns: [feature, bin, left_id, right_id, is_leaf, value]
        - Internal nodes fill columns 0–3 and set is_leaf = 0
        - Leaf nodes fill only value and set is_leaf = 1

        Args:
            tree (list): A list containing a single root node (recursive dict form).
            max_nodes (int): Max number of nodes to allocate in the flat matrix.

        Returns:
            torch.Tensor: [max_nodes x 6] matrix representing the flattened tree.
        """
        flat = torch.full((max_nodes, 6), float("nan"), dtype=torch.float32)
        node_counter = [0]
        node_list = []

        def walk(node):
            curr_id = node_counter[0]
            node_counter[0] += 1

            new_node = {"node_id": curr_id}
            if "leaf_value" in node:
                new_node["leaf_value"] = float(node["leaf_value"])
            else:
                new_node["best_feature"] = float(node["feature"])
                new_node["split_bin"] = float(node["bin"])
                new_node["left_id"] = node_counter[0]
                walk(node["left"])
                new_node["right_id"] = node_counter[0]
                walk(node["right"])

            node_list.append(new_node)
            return new_node

        walk(tree)

        for node in node_list:
            i = node["node_id"]
            if "leaf_value" in node:
                flat[i, 4] = 1.0
                flat[i, 5] = node["leaf_value"]
            else:
                flat[i, 0] = node["best_feature"]
                flat[i, 1] = node["split_bin"]
                flat[i, 2] = node["left_id"]
                flat[i, 3] = node["right_id"]
                flat[i, 4] = 0.0

        return flat
