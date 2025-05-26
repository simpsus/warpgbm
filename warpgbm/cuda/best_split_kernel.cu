#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void directional_split_kernel(
    const float *__restrict__ G,                // [E * F * B]
    const float *__restrict__ H,                // [E * F * B]
    int E, int F, int B,
    float min_split_gain,
    float min_child_samples,
    float eps,
    float *__restrict__ per_era_gain,           // [E * F * (B-1)]
    float *__restrict__ per_era_direction       // [E * F * (B-1)]
)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x; // feature index
    int e = blockIdx.y;                            // era index

    if (f >= F || e >= E) return;

    // Access base offset for this (era, feature)
    int base = e * F * B + f * B;
    int base_gain = e * F * (B - 1) + f * (B - 1);

    float G_total = 0.0f, H_total = 0.0f;
    for (int b = 0; b < B; ++b) {
        G_total += G[base + b];
        H_total += H[base + b];
    }

    float G_L = 0.0f, H_L = 0.0f;
    for (int b = 0; b < B - 1; ++b) {
        G_L += G[base + b];
        H_L += H[base + b];

        float G_R = G_total - G_L;
        float H_R = H_total - H_L;

        float gain = 0.0f;
        float dir = 0.0f;

        if (H_L >= min_child_samples && H_R >= min_child_samples) {
            gain = (G_L * G_L) / (H_L + eps)
                 + (G_R * G_R) / (H_R + eps)
                 - (G_total * G_total) / (H_total + eps);

            float left_val = G_L / (H_L + eps);
            float right_val = G_R / (H_R + eps);
            dir = (left_val > right_val) ? 1.0f : -1.0f;
        }

        per_era_gain[base_gain + b] = gain;
        per_era_direction[base_gain + b] = dir;
    }
}

void launch_directional_split_kernel(
    const at::Tensor &G, // [E, F, B]
    const at::Tensor &H, // [E, F, B]
    float min_split_gain,
    float min_child_samples,
    float eps,
    at::Tensor &per_era_gain,       // [E, F, B]
    at::Tensor &per_era_direction,  // [E, F, B]
    int threads = 128)
{
    int E = G.size(0);
    int F = G.size(1);
    int B = G.size(2);

    dim3 blocks((F + threads - 1) / threads, E); // (feature blocks, era grid)
    dim3 thread_dims(threads);

    directional_split_kernel<<<blocks, thread_dims>>>(
        G.data_ptr<float>(),
        H.data_ptr<float>(),
        E, F, B,
        min_split_gain,
        min_child_samples,
        eps,
        per_era_gain.data_ptr<float>(),
        per_era_direction.data_ptr<float>());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Directional split kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

