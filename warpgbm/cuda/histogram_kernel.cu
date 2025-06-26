#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void histogram_tiled_configurable_kernel(
    const int8_t *__restrict__ bin_indices,
    const float  *__restrict__ residuals,
    const int32_t *__restrict__ sample_indices,
    const int32_t *__restrict__ feature_indices,
    const int32_t *__restrict__ era_indices,
    float *__restrict__ grad_hist,
    float *__restrict__ hess_hist,
    int64_t N, int64_t F_master, int64_t F,
    int64_t B,
    /* NEW */ int64_t era_offset,     // where this tile starts
    /* NEW */ int64_t eras_here,      // size of this tile
    int rows_per_thread)
{
    int hist_feat_idx = blockIdx.x;
    int feat          = feature_indices[hist_feat_idx];

    // -------- shared memory for this tile only --------
    extern __shared__ float shmem[];
    float *sh_grad = shmem;                    // [eras_here * B]
    float *sh_hess = sh_grad + eras_here * B;

    for (int i = threadIdx.x; i < eras_here * B; i += blockDim.x) {
        sh_grad[i] = 0.f;
        sh_hess[i] = 0.f;
    }
    __syncthreads();

    // -------- same row loop as before --------
    int row_start = (blockIdx.y * blockDim.x + threadIdx.x) * rows_per_thread;
    for (int r = 0; r < rows_per_thread; ++r) {
        int row = row_start + r;
        if (row < N) {
            int sample = sample_indices[row];
            int8_t bin = bin_indices[sample * F_master + feat];
            int era    = era_indices[sample];

            // keep rows that fall into this tile
            if (era >= era_offset && era < era_offset + eras_here &&
                bin >= 0 && bin < B)
            {
                int local = era - era_offset;
                atomicAdd(&sh_grad[local * B + bin], residuals[sample]);
                atomicAdd(&sh_hess[local * B + bin], 1.f);
            }
        }
    }
    __syncthreads();

    // -------- flush tile to global --------
    for (int b = threadIdx.x; b < eras_here * B; b += blockDim.x) {
        int local   = b / B;
        int bin     = b % B;
        int globalE = era_offset + local;

        size_t idx = (size_t)globalE * F * B + hist_feat_idx * B + bin;
        atomicAdd(&grad_hist[idx], sh_grad[b]);
        atomicAdd(&hess_hist[idx], sh_hess[b]);
    }
}


void launch_histogram_kernel_cuda_configurable(
  const at::Tensor &bin_indices,
    const at::Tensor &residuals,
    const at::Tensor &sample_indices,
    const at::Tensor &feature_indices,
    const at::Tensor &era_indices,
    at::Tensor &grad_hist,
    at::Tensor &hess_hist,
    int num_bins,
    int threads_per_block = 256,
    int rows_per_thread = 1
)
{
    const int ERA_TILE = 64;                    // fits 48 KB up to B≈350

    int64_t N  = sample_indices.size(0);
    int64_t F  = feature_indices.size(0);
    int num_features_master = bin_indices.size(1);
    int num_eras = grad_hist.size(0);

    int64_t rows_per_block = threads_per_block * rows_per_thread;
    int64_t row_tiles      = (N + rows_per_block - 1) / rows_per_block;

    dim3 blocks(F, row_tiles);                 // same as before
    dim3 threads(threads_per_block);

    // allow 96 KB if the GPU supports it
    int max_dyn_smem;
    cudaDeviceGetAttribute(&max_dyn_smem,
        cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    cudaFuncSetAttribute(histogram_tiled_configurable_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        max_dyn_smem);

    // ---- launch once per era-tile ----
    for (int era_offset = 0; era_offset < num_eras; era_offset += ERA_TILE)
    {
        int eras_here = std::min(ERA_TILE, num_eras - era_offset);
        int shared_mem_bytes = 2ULL * eras_here * num_bins * sizeof(float);

        histogram_tiled_configurable_kernel<<<blocks, threads,
                                              shared_mem_bytes>>>(
            bin_indices.data_ptr<int8_t>(),
            residuals.data_ptr<float>(),
            sample_indices.data_ptr<int32_t>(),
            feature_indices.data_ptr<int32_t>(),
            era_indices.data_ptr<int32_t>(),
            grad_hist.data_ptr<float>(),
            hess_hist.data_ptr<float>(),
            N, num_features_master, F, num_bins,
            era_offset, eras_here,          // NEW args
            rows_per_thread);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
}

