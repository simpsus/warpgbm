#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void best_split_kernel_global_only(
    const float *__restrict__ G, // [F x B]
    const float *__restrict__ H, // [F x B]
    int F,
    int B,
    float min_split_gain,
    float min_child_samples,
    float eps,
    float *__restrict__ best_gains, // [F]
    int *__restrict__ best_bins     // [F]
)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= F)
        return;

    float G_total = 0.0f, H_total = 0.0f;
    for (int b = 0; b < B; ++b)
    {
        G_total += G[f * B + b];
        H_total += H[f * B + b];
    }

    float G_L = 0.0f, H_L = 0.0f;
    float best_gain = min_split_gain;
    int best_bin = -1;

    for (int b = 0; b < B - 1; ++b)
    {
        G_L += G[f * B + b];
        H_L += H[f * B + b];
        float G_R = G_total - G_L;
        float H_R = H_total - H_L;

        if (H_L >= min_child_samples && H_R >= min_child_samples)
        {
            float gain = (G_L * G_L) / (H_L + eps) + (G_R * G_R) / (H_R + eps);
            if (gain > best_gain)
            {
                best_gain = gain;
                best_bin = b;
            }
        }
    }

    best_gains[f] = best_gain;
    best_bins[f] = best_bin;
}

void launch_best_split_kernel_cuda(
    const at::Tensor &G, // [F x B]
    const at::Tensor &H, // [F x B]
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor &best_gains, // [F], float32
    at::Tensor &best_bins,  // [F], int32
    int threads)
{
    int F = G.size(0);
    int B = G.size(1);

    int blocks = (F + threads - 1) / threads;

    best_split_kernel_global_only<<<blocks, threads>>>(
        G.data_ptr<float>(),
        H.data_ptr<float>(),
        F,
        B,
        min_split_gain,
        min_child_samples,
        eps,
        best_gains.data_ptr<float>(),
        best_bins.data_ptr<int>());
}
