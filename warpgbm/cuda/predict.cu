#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void predict_forest_kernel(
    const int8_t *__restrict__ bin_indices, // [N x F]
    const float *__restrict__ tree_tensor,  // [T x max_nodes x 6]
    int64_t N, int64_t F, int64_t T, int64_t max_nodes,
    float learning_rate,
    float *__restrict__ out // [N]
)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_jobs = N * T;
    if (idx >= total_jobs)
        return;

    int64_t i = idx % N; // sample index
    int64_t t = idx / N; // tree index

    const float *tree = tree_tensor + t * max_nodes * 6;

    int node_id = 0;
    while (true)
    {
        float is_leaf = tree[node_id * 6 + 4];
        if (is_leaf > 0.5f)
        {
            float val = tree[node_id * 6 + 5];
            atomicAdd(&out[i], learning_rate * val);
            return;
        }

        int feat = static_cast<int>(tree[node_id * 6 + 0]);
        int split_bin = static_cast<int>(tree[node_id * 6 + 1]);
        int left_id = static_cast<int>(tree[node_id * 6 + 2]);
        int right_id = static_cast<int>(tree[node_id * 6 + 3]);

        // prevent overflow
        int64_t bin_idx = i * F + feat;
        int8_t bin = bin_indices[bin_idx];

        node_id = (bin <= split_bin) ? left_id : right_id;
    }
}


void predict_with_forest(
    const at::Tensor &bin_indices,
    const at::Tensor &tree_tensor,
    float learning_rate,
    at::Tensor &out
)
{
    int64_t N = bin_indices.size(0);
    int64_t F = bin_indices.size(1);
    int64_t T = tree_tensor.size(0);
    int64_t max_nodes = tree_tensor.size(1);

    int64_t total_jobs = N * T;
    int threads_per_block = 256;
    int64_t blocks = (total_jobs + threads_per_block - 1) / threads_per_block;

    predict_forest_kernel<<<blocks, threads_per_block>>>(
        bin_indices.data_ptr<int8_t>(),
        tree_tensor.data_ptr<float>(),
        N, F, T, max_nodes,
        learning_rate,
        out.data_ptr<float>());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA predict kernel failed: %s\n", cudaGetErrorString(err));
    }
}
