#include "BatchedDesignMatrix.hpp"
#include "BinSpMV.hpp"
#include <NumCpp.hpp>
#include <memory>

#define WARPSIZE 32

__global__ void binspmv_kernel(int num_rows, size_t *row_ptrs, size_t *col_indices,
                               float *x, float *y) {

  __shared__ float partial_sums[WARPSIZE];

  size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t row_idx = thread_idx / WARPSIZE;
  size_t lane_idx = thread_idx % WARPSIZE;

  if (row_idx < num_rows) {
    size_t row_start = row_ptrs[row_idx];
    size_t row_end = row_ptrs[row_idx + 1];

    partial_sums[lane_idx] = 0;
    for (int j = row_start + lane_idx; j < row_end; j += WARPSIZE) {
      partial_sums[lane_idx] += x[col_indices[j]];
    }
  }

  if (lane_idx == 0) {
    float res = 0;
    for (int lane_idx = 0; lane_idx < WARPSIZE; lane_idx++) {
      res += partial_sums[lane_idx];
    }

    y[row_idx] = res;
  }
}

std::unique_ptr<nc::NdArray<float>> batch_binspmv(BatchedDesignMatrix& A, const nc::NdArray<float>& x) {
  assert(x.shape().cols == 1);
  size_t num_batchs = A.len();
  size_t batch_size = A.batch_size();

  auto res = std::make_unique<nc::NdArray<float>>(A.nrow(), 1); 

  for (int batch_idx = 0; batch_idx < num_batchs; batch_idx++) {
    auto batched_A = A.get(batch_idx);
    auto num_rows = batched_A->nrow();
    const auto& row_ptrs = batched_A->row_ptrs;  
    const auto& col_indices = batched_A->col_indices;  
    
    size_t *row_ptrs_d, *col_indices_d;
    float *x_d, *y_d;
    auto size_row_ptrs = row_ptrs.size() * sizeof(size_t);
    auto size_col_indices = col_indices.size() * sizeof(size_t);
    auto size_x = x.shape().rows * sizeof(float);
    auto size_y = num_rows * sizeof(float);

    cudaMalloc(&row_ptrs_d, size_row_ptrs);
    cudaMalloc(&col_indices_d, size_col_indices);
    cudaMalloc(&x_d, size_x);
    cudaMalloc(&y_d, size_y);
  
    cudaMemcpy(row_ptrs_d, row_ptrs.data(), size_row_ptrs, cudaMemcpyHostToDevice);
    cudaMemcpy(col_indices_d, col_indices.data(), size_col_indices, cudaMemcpyHostToDevice);
    cudaMemcpy(x_d, x.data(), size_x, cudaMemcpyHostToDevice);

    binspmv_kernel<<<num_rows, WARPSIZE>>>(num_rows, row_ptrs_d, col_indices_d, x_d, y_d);

    A.prefetch((batch_idx + 1) % num_batchs);

    cudaDeviceSynchronize();
    auto batch_start = batch_idx * batch_size;
    cudaMemcpy(res->data() + batch_start, y_d, num_rows * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(row_ptrs_d);
    cudaFree(col_indices_d);
    cudaFree(x_d);
    cudaFree(y_d);
  }

  return res;
}
