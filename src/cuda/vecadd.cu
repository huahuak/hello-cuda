#include "vecadd.cuh"

__global__ void i_vector_add(float *out, float *a, float *b, int n) {
  int index = threadIdx.x;
  int stride = blockDim.x;

  for (int i = index; i < n; i += stride) {
    out[i] = a[i] + b[i];
  }
}

namespace cu {
void vector_add(float *out, float *a, float *b, int n) {
  i_vector_add<<<1,256>>>(out, a, b, n);
}
} // namespace cu