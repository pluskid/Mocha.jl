#define THREADS_PER_BLOCK 1024

#include "logistic_loss.impl"

extern "C" {
__global__ void test(int n, float *input, float *output) {
  __shared__ float local_mem[THREADS_PER_BLOCK];
  
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= n) {
    local_mem[idx] = 0;
    return;
  }

  local_mem[idx] = -log(*input);

  __syncthreads();
  if (0 == threadIdx.x) {
    // thread 0 does in-block accumulation
    float total_local_loss = 0;
    for (int i = 0; i < THREADS_PER_BLOCK; ++i)
      total_local_loss += local_mem[i];
    atomicAdd(output, static_cast<float>(total_local_loss));
  }
}
}
