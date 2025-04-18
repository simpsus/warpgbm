import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from warpgbm.cuda import node_kernel
from tqdm import tqdm
from typing import Tuple
from torch import Tensor

histogram_kernels = {
    'hist1': node_kernel.compute_histogram,
    'hist2': node_kernel.compute_histogram2,
    'hist3': node_kernel.compute_histogram3
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
        histogram_computer='hist3',
        threads_per_block=64,
        rows_per_thread=4,
        L2_reg = 1e-6,
        L1_reg = 0.0,
        device = 'cuda'
    ):
        self.num_bins = num_bins
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.forest = None
        self.bin_edges = None  # shape: [num_features, num_bins-1] if using quantile binning
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

    def fit(self, X, y, era_id=None):
        if era_id is None:
            era_id = np.ones(X.shape[0], dtype='int32')
        self.bin_indices, era_indices, self.bin_edges, self.unique_eras, self.Y_gpu = self.preprocess_gpu_data(X, y, era_id)
        self.num_samples, self.num_features = X.shape
        self.gradients = torch.zeros_like(self.Y_gpu)
        self.root_node_indices = torch.arange(self.num_samples, device=self.device)
        self.base_prediction = self.Y_gpu.mean().item()
        self.gradients += self.base_prediction
        self.best_gains = torch.zeros(self.num_features, device=self.device)
        self.best_bins = torch.zeros(self.num_features, device=self.device, dtype=torch.int32)
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
                    print("Detected pre-binned integer input — skipping quantile binning.")
                    bin_indices = torch.from_numpy(X_np).to(self.device).contiguous().to(torch.int8)
        
                    # We'll store None or an empty tensor in self.bin_edges
                    # to indicate that we skip binning at predict-time
                    bin_edges = torch.arange(1, self.num_bins, dtype=torch.float32).repeat(self.num_features, 1)
                    bin_edges = bin_edges.to(self.device)
                    unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)
                    return bin_indices, era_indices, bin_edges, unique_eras, Y_gpu
                else:
                    print("Integer input detected, but values exceed num_bins — falling back to quantile binning.")
        
            bin_indices = torch.empty((self.num_samples, self.num_features), dtype=torch.int8, device='cuda')
            bin_edges = torch.empty((self.num_features, self.num_bins - 1), dtype=torch.float32, device='cuda')

            X_np = torch.from_numpy(X_np).to(torch.float32).pin_memory()

            for f in range(self.num_features):
                X_f = X_np[:, f].to('cuda', non_blocking=True)
                quantiles = torch.linspace(0, 1, self.num_bins + 1, device='cuda', dtype=X_f.dtype)[1:-1]
                bin_edges_f = torch.quantile(X_f, quantiles, dim=0).contiguous()  # shape: [B-1] for 1D input
                bin_indices_f = bin_indices[:, f].contiguous()  # view into output
                node_kernel.custom_cuda_binner(X_f, bin_edges_f, bin_indices_f)
                bin_indices[:,f] = bin_indices_f
                bin_edges[f,:] = bin_edges_f

            unique_eras, era_indices = torch.unique(era_id_gpu, return_inverse=True)
            return bin_indices, era_indices, bin_edges, unique_eras, Y_gpu

    def compute_histograms(self, bin_indices_sub, gradients):
        grad_hist = torch.zeros((self.num_features, self.num_bins), device=self.device, dtype=torch.float32)
        hess_hist = torch.zeros((self.num_features, self.num_bins), device=self.device, dtype=torch.float32)
    
        self.compute_histogram(
            bin_indices_sub,
            gradients,
            grad_hist,
            hess_hist,
            self.num_bins,
            self.threads_per_block,
            self.rows_per_thread
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
            self.threads_per_block
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
        best_feature, best_bin = self.find_best_split(gradient_histogram, hessian_histogram)
    
        if best_feature == -1:
            leaf_value = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * leaf_value
            return {"leaf_value": leaf_value.item(), "samples": parent_size}
    
        split_mask = (self.bin_indices[node_indices, best_feature] <= best_bin)
        left_indices = node_indices[split_mask]
        right_indices = node_indices[~split_mask]

        left_size = left_indices.numel()
        right_size = right_indices.numel()

        if left_size == 0 or right_size == 0:
            leaf_value = self.residual[node_indices].mean()
            self.gradients[node_indices] += self.learning_rate * leaf_value
            return {"leaf_value": leaf_value.item(), "samples": parent_size}

        if left_size <= right_size:
            grad_hist_left, hess_hist_left = self.compute_histograms( self.bin_indices[left_indices], self.residual[left_indices] )
            grad_hist_right = gradient_histogram - grad_hist_left
            hess_hist_right = hessian_histogram - hess_hist_left
        else:
            grad_hist_right, hess_hist_right = self.compute_histograms( self.bin_indices[right_indices], self.residual[right_indices] )
            grad_hist_left = gradient_histogram - grad_hist_right
            hess_hist_left = hessian_histogram - hess_hist_right

        new_depth = depth + 1
        left_child = self.grow_tree(grad_hist_left, hess_hist_left, left_indices, new_depth)
        right_child = self.grow_tree(grad_hist_right, hess_hist_right, right_indices, new_depth)
    
        return { "feature": best_feature, "bin": best_bin, "left": left_child, "right": right_child }

    def grow_forest(self):
        forest = [{} for _ in range(self.n_estimators)]
        self.training_loss = []
    
        for i in tqdm( range(self.n_estimators) ):
            self.residual = self.Y_gpu - self.gradients
    
            self.root_gradient_histogram, self.root_hessian_histogram = \
                self.compute_histograms(self.bin_indices, self.residual)
    
            tree = self.grow_tree(
                self.root_gradient_histogram,
                self.root_hessian_histogram,
                self.root_node_indices,
                depth=0
            )
            forest[i] = tree
        # loss = ((self.Y_gpu - self.gradients) ** 2).mean().item()
        # self.training_loss.append(loss)
        # print(f"🌲 Tree {i+1}/{self.n_estimators} - MSE: {loss:.6f}")

        print("Finished training forest.")
        return forest

    def predict(self, X_np, chunk_size=50000):
        """
        Vectorized predict using a padded layer-by-layer approach.
        We assume `flatten_forest_to_tensors` has produced self.flat_forest with
        "features", "thresholds", "leaf_values", all shaped [n_trees, max_nodes].
        """
        with torch.no_grad():
            # 1) Convert X_np -> bin_indices
            is_integer_type = np.issubdtype(X_np.dtype, np.integer)
            if is_integer_type:
                max_vals = X_np.max(axis=0)
                if np.all(max_vals < self.num_bins):
                    bin_indices = X_np.astype(np.int8)
                else:
                    raise ValueError("Pre-binned integers must be < num_bins")
            else:
                X_cpu = torch.from_numpy(X_np).type(torch.float32)
                bin_indices = torch.empty((X_np.shape[0], X_np.shape[1]), dtype=torch.int8)
                bin_edges_cpu = self.bin_edges.to('cpu')
                for f in range(self.num_features):
                    bin_indices[:, f] = torch.bucketize(X_cpu[:, f], bin_edges_cpu[f], right=False).type(torch.int8)
                bin_indices = bin_indices.numpy()

            # 2) Ensure we have a padded representation
            self.flat_forest = self.flatten_forest_to_tensors(self.forest)

            features_t   = self.flat_forest["features"]      # [n_trees, max_nodes], int16
            thresholds_t = self.flat_forest["thresholds"]    # [n_trees, max_nodes], int16
            values_t     = self.flat_forest["leaf_values"]    # [n_trees, max_nodes], float32
            max_nodes    = self.flat_forest["max_nodes"]

            n_trees = features_t.shape[0]
            N       = bin_indices.shape[0]
            out     = np.zeros(N, dtype=np.float32)

            # 3) Process rows in chunks
            for start in tqdm(range(0, N, chunk_size)):
                end = min(start + chunk_size, N)
                chunk_np  = bin_indices[start:end]  # shape [chunk_size, F]
                chunk_gpu = torch.from_numpy(chunk_np).to(self.device)  # [chunk_size, F], int8

                # Accumulate raw (unscaled) leaf sums
                chunk_preds = torch.zeros((end - start,), dtype=torch.float32, device=self.device)

                # node_idx[i] tracks the current node index in the padded tree for row i
                node_idx = torch.zeros((end - start,), dtype=torch.int32, device=self.device)

                # 'active' is a boolean mask over [0..(end-start-1)], indicating which rows haven't reached a leaf
                active = torch.ones((end - start,), dtype=torch.bool, device=self.device)

                for t in range(n_trees):
                    # Reset for each tree (each tree is independent)
                    node_idx.fill_(0)
                    active.fill_(True)

                    tree_features = features_t[t]     # shape [max_nodes], int16
                    tree_thresh   = thresholds_t[t]    # shape [max_nodes], int16
                    tree_values   = values_t[t]          # shape [max_nodes], float32

                    # Up to self.max_depth+1 layers
                    for _level in range(self.max_depth + 1):
                        active_idx = active.nonzero(as_tuple=True)[0]
                        if active_idx.numel() == 0:
                            break  # all rows are done in this tree

                        current_node_idx = node_idx[active_idx]
                        f    = tree_features[current_node_idx]    # shape [#active], int16
                        thr  = tree_thresh[current_node_idx]       # shape [#active], int16
                        vals = tree_values[current_node_idx]       # shape [#active], float32

                        mask_no_node = (f == -2)
                        mask_leaf    = (f == -1)

                        # If leaf, add leaf value and mark inactive.
                        if mask_leaf.any():
                            leaf_rows = active_idx[mask_leaf]
                            chunk_preds[leaf_rows] += vals[mask_leaf]
                            active[leaf_rows] = False

                        # If no node, mark inactive.
                        if mask_no_node.any():
                            no_node_rows = active_idx[mask_no_node]
                            active[no_node_rows] = False

                        # For internal nodes, perform bin comparison.
                        mask_internal = (~mask_leaf & ~mask_no_node)
                        if mask_internal.any():
                            internal_rows = active_idx[mask_internal]
                            act_f   = f[mask_internal].long()
                            act_thr = thr[mask_internal]
                            binvals = chunk_gpu[internal_rows, act_f]
                            go_left = (binvals <= act_thr)
                            new_left_idx  = current_node_idx[mask_internal] * 2 + 1
                            new_right_idx = current_node_idx[mask_internal] * 2 + 2
                            node_idx[internal_rows[go_left]]  = new_left_idx[go_left]
                            node_idx[internal_rows[~go_left]] = new_right_idx[~go_left]
                    # end per-tree layer loop
                # end for each tree

                out[start:end] = (
                    self.base_prediction + self.learning_rate * chunk_preds
                ).cpu().numpy()

            return out

    def flatten_forest_to_tensors(self, forest):
        """
        Convert a list of dict-based trees into a fixed-size array representation
        for each tree, up to max_depth. Each tree is stored in a 'perfect binary tree'
        layout:
          - node 0 is the root
          - node i has children (2*i + 1) and (2*i + 2), if within range
          - feature = -2 indicates no node / invalid
          - feature = -1 indicates a leaf node
          - otherwise, an internal node with that feature.
        """
        n_trees = len(forest)
        max_nodes = 2 ** (self.max_depth + 1) - 1  # total array slots per tree

        # Allocate padded arrays (on CPU for ease of indexing).
        feat_arr = np.full((n_trees, max_nodes), -2, dtype=np.int16)
        thresh_arr = np.full((n_trees, max_nodes), -2, dtype=np.int16)
        value_arr = np.zeros((n_trees, max_nodes), dtype=np.float32)

        def fill_padded(tree, tree_idx, node_idx, depth):
            """
            Recursively fill feat_arr, thresh_arr, value_arr for a single tree.
            If depth == self.max_depth, no children are added.
            If there's no node, feature remains -2.
            """
            if "leaf_value" in tree:
                feat_arr[tree_idx, node_idx] = -1
                thresh_arr[tree_idx, node_idx] = -1
                value_arr[tree_idx, node_idx] = tree["leaf_value"]
                return

            feat = tree["feature"]
            bin_th = tree["bin"]

            feat_arr[tree_idx, node_idx] = feat
            thresh_arr[tree_idx, node_idx] = bin_th
            # Internal nodes keep a 0 value.

            if depth < self.max_depth:
                left_idx  = 2 * node_idx + 1
                right_idx = 2 * node_idx + 2
                fill_padded(tree["left"],  tree_idx, left_idx, depth + 1)
                fill_padded(tree["right"], tree_idx, right_idx, depth + 1)
            # At max depth, children remain unfilled (-2).

        for t, root in enumerate(forest):
            fill_padded(root, t, 0, 0)

        # Convert to torch Tensors on the proper device.
        features_t = torch.from_numpy(feat_arr).to(self.device)
        thresholds_t = torch.from_numpy(thresh_arr).to(self.device)
        leaf_values_t = torch.from_numpy(value_arr).to(self.device)

        return {
            "features": features_t,       # [n_trees, max_nodes]
            "thresholds": thresholds_t,     # [n_trees, max_nodes]
            "leaf_values": leaf_values_t,   # [n_trees, max_nodes]
            "max_nodes": max_nodes
        }

    def predict_numpy(self, X_np, chunk_size=50000):
        """
        Fully NumPy-based version of predict_fast.
        Assumes flatten_forest_to_tensors has been called and `self.flat_forest` is ready.
        """
        # 1) Convert X_np -> bin_indices
        is_integer_type = np.issubdtype(X_np.dtype, np.integer)
        if is_integer_type:
            max_vals = X_np.max(axis=0)
            if np.all(max_vals < self.num_bins):
                bin_indices = X_np.astype(np.int8)
            else:
                raise ValueError("Pre-binned integers must be < num_bins")
        else:
            bin_indices = np.empty_like(X_np, dtype=np.int8)
            # Ensure bin_edges are NumPy arrays
            if isinstance(self.bin_edges[0], torch.Tensor):
                bin_edges_np = [be.cpu().numpy() for be in self.bin_edges]
            else:
                bin_edges_np = self.bin_edges

            for f in range(self.num_features):
                bin_indices[:, f] = np.searchsorted(bin_edges_np[f], X_np[:, f], side='left')

        # Ensure we have a padded representation
        self.flat_forest = self.flatten_forest(self.forest)

        # 2) Padded forest arrays (already NumPy now)
        features_t   = self.flat_forest["features"]      # [n_trees, max_nodes], int16
        thresholds_t = self.flat_forest["thresholds"]    # [n_trees, max_nodes], int16
        values_t     = self.flat_forest["leaf_values"]   # [n_trees, max_nodes], float32
        max_nodes    = self.flat_forest["max_nodes"]
        n_trees      = features_t.shape[0]
        N            = bin_indices.shape[0]
        out          = np.zeros(N, dtype=np.float32)

        # 3) Process in chunks
        for start in tqdm( range(0, N, chunk_size) ):
            end = min(start + chunk_size, N)
            chunk = bin_indices[start:end]  # [chunk_size, F]
            chunk_preds = np.zeros(end - start, dtype=np.float32)

            for t in range(n_trees):
                node_idx = np.zeros(end - start, dtype=np.int32)
                active = np.ones(end - start, dtype=bool)

                tree_features = features_t[t]   # [max_nodes]
                tree_thresh   = thresholds_t[t] # [max_nodes]
                tree_values   = values_t[t]     # [max_nodes]

                for _level in range(self.max_depth + 1):
                    active_idx = np.nonzero(active)[0]
                    if active_idx.size == 0:
                        break

                    current_node_idx = node_idx[active_idx]
                    f    = tree_features[current_node_idx]
                    thr  = tree_thresh[current_node_idx]
                    vals = tree_values[current_node_idx]

                    mask_no_node = (f == -2)
                    mask_leaf    = (f == -1)
                    mask_internal = ~(mask_leaf | mask_no_node)

                    if np.any(mask_leaf):
                        leaf_rows = active_idx[mask_leaf]
                        chunk_preds[leaf_rows] += vals[mask_leaf]
                        active[leaf_rows] = False

                    if np.any(mask_no_node):
                        no_node_rows = active_idx[mask_no_node]
                        active[no_node_rows] = False

                    if np.any(mask_internal):
                        internal_rows = active_idx[mask_internal]
                        act_f   = f[mask_internal].astype(np.int32)
                        act_thr = thr[mask_internal]
                        binvals = chunk[internal_rows, act_f]
                        go_left = binvals <= act_thr

                        new_left_idx  = current_node_idx[mask_internal] * 2 + 1
                        new_right_idx = current_node_idx[mask_internal] * 2 + 2
                        node_idx[internal_rows[go_left]]  = new_left_idx[go_left]
                        node_idx[internal_rows[~go_left]] = new_right_idx[~go_left]

            out[start:end] = self.base_prediction + self.learning_rate * chunk_preds

        return out

    def flatten_forest(self, forest):
        n_trees = len(forest)
        max_nodes = 2 ** (self.max_depth + 1) - 1

        feat_arr = np.full((n_trees, max_nodes), -2, dtype=np.int16)
        thresh_arr = np.full((n_trees, max_nodes), -2, dtype=np.int16)
        value_arr = np.zeros((n_trees, max_nodes), dtype=np.float32)

        def fill_padded(tree, tree_idx, node_idx, depth):
            if "leaf_value" in tree:
                feat_arr[tree_idx, node_idx] = -1
                thresh_arr[tree_idx, node_idx] = -1
                value_arr[tree_idx, node_idx] = tree["leaf_value"]
                return
            feat = tree["feature"]
            bin_th = tree["bin"]
            feat_arr[tree_idx, node_idx] = feat
            thresh_arr[tree_idx, node_idx] = bin_th

            if depth < self.max_depth:
                left_idx  = 2 * node_idx + 1
                right_idx = 2 * node_idx + 2
                fill_padded(tree["left"],  tree_idx, left_idx, depth + 1)
                fill_padded(tree["right"], tree_idx, right_idx, depth + 1)

        for t, root in enumerate(forest):
            fill_padded(root, t, 0, 0)

        return {
            "features": feat_arr,
            "thresholds": thresh_arr,
            "leaf_values": value_arr,
            "max_nodes": max_nodes
        }
