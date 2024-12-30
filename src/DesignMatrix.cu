#include "DesignMatrix.hpp"
#include "omp.h"
#include <NumCpp.hpp>
#include <NumCpp/Functions/logical_or.hpp>
#include <NumCpp/Functions/where.hpp>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sys/types.h>

#define WARPSIZE 32

#define KERNEL(NAME, FIRST, SECOND)                                            \
  __global__ void NAME(size_t row_start, size_t row_end, size_t col_start,     \
                       size_t col_end, float *x, float *y, float *dataframe,   \
                       size_t *interaction, size_t *len_interact,              \
                       size_t *sample_idx, size_t df_ncol, size_t max_order) { \
                                                                               \
    __shared__ float partial_sums[WARPSIZE];                                   \
                                                                               \
    size_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;                 \
    size_t warp_idx = thread_idx / WARPSIZE;                                   \
    size_t lane_idx = thread_idx % WARPSIZE;                                   \
                                                                               \
    auto FIRST##_idx = FIRST##_start + warp_idx;                               \
                                                                               \
    if (FIRST##_idx < FIRST##_end) {                                           \
      partial_sums[lane_idx] = 0;                                              \
      for (int SECOND##_idx = SECOND##_start + lane_idx;                       \
           SECOND##_idx < SECOND##_end; SECOND##_idx += WARPSIZE) {            \
        size_t cur_sample_idx = sample_idx[col_idx];                           \
        size_t cur_len_interact = len_interact[col_idx];                       \
        bool nonzero = true;                                                   \
                                                                               \
        for (int j = 0; j < cur_len_interact; j++) {                           \
          size_t cur_interact = interaction[col_idx * max_order + j];          \
          float thres = dataframe[cur_sample_idx * df_ncol + cur_interact];    \
          float val = dataframe[row_idx * df_ncol + cur_interact];             \
          nonzero &= (val >= thres);                                           \
        }                                                                      \
                                                                               \
        if (nonzero) {                                                         \
          partial_sums[lane_idx] += x[SECOND##_idx - SECOND##_start];          \
        }                                                                      \
      }                                                                        \
    }                                                                          \
                                                                               \
    if (lane_idx == 0) {                                                       \
      float res = 0;                                                           \
      for (int lane_idx = 0; lane_idx < WARPSIZE; lane_idx++) {                \
        res += partial_sums[lane_idx];                                         \
      }                                                                        \
      y[FIRST##_idx - FIRST##_start] = res;                                    \
    }                                                                          \
  }

KERNEL(fusedRegionMV_kernel, row, col);
KERNEL(fusedRegionVM_kernel, col, row);

std::unique_ptr<nc::NdArray<float>>
DesignMatrix::fusedRegionMV(size_t row_start, size_t row_end, size_t col_start,
                            size_t col_end, const nc::NdArray<float> &x,
                            bool transpose) const {

  bool valid_row_bounds = (row_end <= this->_nrow) && (row_start <= row_end);
  bool valid_col_bounds = (col_end <= this->_ncol) && (col_start <= col_end);

  if (!valid_row_bounds || !valid_col_bounds) {
    throw std::out_of_range("Index out of range");
  }

  auto shifted_row_start = row_start + this->_offset;
  auto shifted_row_end = row_end + this->_offset;

  size_t res_len;
  if (!transpose) {
    res_len = row_end - row_start;
  } else {
    res_len = col_end - col_start;
  }

  auto res = std::make_unique<nc::NdArray<float>>(res_len, 1);

  float *x_cuda;
  float *y_cuda;
  auto size_x = x.shape().rows * sizeof(float);
  auto size_y = res_len * sizeof(float);
  cudaMalloc(&x_cuda, size_x);
  cudaMalloc(&y_cuda, size_y);
  cudaMemcpy(x_cuda, x.data(), size_x, cudaMemcpyHostToDevice);

  if (!transpose) {
    fusedRegionMV_kernel<<<res_len, WARPSIZE>>>(
        shifted_row_start, shifted_row_end, col_start, col_end, x_cuda, y_cuda,
        this->_dataframe_cuda, this->_interaction_cuda,
        this->_len_interact_cuda, this->_sample_idx_cuda,
        this->_dataframe.shape().cols, this->_max_order);
  } else {
    fusedRegionVM_kernel<<<res_len, WARPSIZE>>>(
        shifted_row_start, shifted_row_end, col_start, col_end, x_cuda, y_cuda,
        this->_dataframe_cuda, this->_interaction_cuda,
        this->_len_interact_cuda, this->_sample_idx_cuda,
        this->_dataframe.shape().cols, this->_max_order);
  }

  cudaDeviceSynchronize();
  cudaMemcpy(res->data(), y_cuda, res_len * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaFree(x_cuda);
  cudaFree(y_cuda);

  return res;
}

void DesignMatrix::_init_ColIndices(size_t order, int prev_idx,
                                    std::vector<size_t> &interact) {
  if (order == 0) {
    auto num_sampled_row = this->_sampled_row.shape().cols;
    for (int i = 0; i < num_sampled_row; i++) {
      int row_idx = this->_sampled_row(0, i);
      this->ColIndices.push_back(
          ColIndex{interact, static_cast<size_t>(row_idx)});
    }
    return;
  }

  size_t df_ncol = this->_dataframe.shape().cols;
  for (int i = prev_idx + 1; i < df_ncol; i++) {
    interact.push_back(i);
    this->_init_ColIndices(order - 1, i, interact);
    interact.pop_back();
  }
}

DesignMatrix::DesignMatrix(const nc::NdArray<float> &dataframe,
                           size_t max_order, float sample_ratio, float reduce_epsilon)
    : _dataframe(dataframe), _max_order(max_order), _type("train"), _offset(0) {
  auto df_shape = dataframe.shape();
  size_t df_nrow = df_shape.rows;
  this->_nrow = df_nrow;

  if (sample_ratio == 1) {
    this->_sampled_row =
        nc::arange<int>(0, this->_nrow).reshape(1, this->_nrow);
  } else {
    uint32_t num_sampled_row = std::floor(this->_nrow * sample_ratio);
    this->_sampled_row =
        nc::random::randInt<int>({1, num_sampled_row}, 0, this->_nrow);
  }
  for (int o = 1; o <= max_order; o++) {
    std::vector<size_t> interact;
    this->_init_ColIndices(o, -1, interact);
  }

  this->_ncol = this->ColIndices.size();
  this->_allocate_cudamem();

  if (reduce_epsilon != -1) {
    this->reduce_basis(reduce_epsilon);
  }
};

DesignMatrix::DesignMatrix(const DesignMatrix &other) {
  this->_dataframe = other._dataframe;
  this->_type = other._type;
  this->_max_order = other._max_order;
  this->_nrow = other._nrow;
  this->_ncol = other._ncol;
  this->_offset = other._offset;
  this->_sampled_row = other._sampled_row;
  for (auto& c : other.ColIndices) {
    this->ColIndices.push_back({c.interaction, c.sample_idx});
  }
};

DesignMatrix::~DesignMatrix() {
  cudaFree(_dataframe_cuda);
  cudaFree(_interaction_cuda);
  cudaFree(_len_interact_cuda);
  cudaFree(_sample_idx_cuda);
}

void DesignMatrix::_init_PredDesignMatrix(const nc::NdArray<float> &new_df) {
  this->_offset = this->_dataframe.shape().rows;
  this->_dataframe = nc::stack({this->_dataframe, new_df}, nc::Axis::ROW);
  this->_type = "prediction";
  this->_nrow = new_df.shape().rows;
  this->_allocate_cudamem();
}

std::unique_ptr<DesignMatrix>
DesignMatrix::getPredDesignMatrix(const nc::NdArray<float> &new_df) const {
  if (this->_type != "train") {
    std::cerr << "Should be called from DesignMatrix with type `train`"
              << std::endl;
  }
  auto res = std::make_unique<DesignMatrix>(*this);
  res->_init_PredDesignMatrix(new_df);
  return res;
}

void DesignMatrix::_allocate_cudamem(bool reserve_df) {
  auto df_shape = this->_dataframe.shape();
  size_t df_nrow = df_shape.rows;
  size_t df_ncol = df_shape.cols;

  auto size_df = df_nrow * df_ncol * sizeof(float);
  auto size_interact = this->_ncol * this->_max_order * sizeof(size_t);
  auto size_len_interat = this->_ncol * sizeof(size_t);
  auto size_sample_idx = this->_ncol * sizeof(size_t);

  if (!reserve_df) {
    cudaMalloc(&this->_dataframe_cuda, size_df);
  }
  cudaMalloc(&this->_interaction_cuda, size_interact);
  cudaMalloc(&this->_len_interact_cuda, size_len_interat);
  cudaMalloc(&this->_sample_idx_cuda, size_sample_idx);

  size_t *arr_interaction = (size_t *)malloc(size_interact);
  size_t *arr_len_interact = (size_t *)malloc(size_len_interat);
  size_t *arr_sample_idx = (size_t *)malloc(size_len_interat);

  /* TODO: Can be parallelize */
  for (int i = 0; i < this->_ncol; i++) {
    auto &col_index = this->ColIndices[i];
    auto &interact = col_index.interaction;
    auto sample_idx = col_index.sample_idx;

    auto cur_len_interact = interact.size();
    arr_len_interact[i] = cur_len_interact;
    arr_sample_idx[i] = sample_idx;

    for (int j = 0; j < cur_len_interact; j++) {
      arr_interaction[i * this->_max_order + j] = interact[j];
    }
  }

  if (!reserve_df) {
    cudaMemcpy(this->_dataframe_cuda, this->_dataframe.data(), size_df,
              cudaMemcpyHostToDevice);
  }
  cudaMemcpy(this->_interaction_cuda, arr_interaction, size_interact,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->_len_interact_cuda, arr_len_interact, size_len_interat,
             cudaMemcpyHostToDevice);
  cudaMemcpy(this->_sample_idx_cuda, arr_sample_idx, size_sample_idx,
             cudaMemcpyHostToDevice);

  free(arr_interaction);
  free(arr_len_interact);
  free(arr_sample_idx);
}

std::unique_ptr<nc::NdArray<float>> DesignMatrix::proportion_ones() const {
  auto proportion_ones = this->fusedRegionMV(
    0,
    this->_nrow,
    0,
    this->_ncol,
    nc::ones<float>(this->_nrow, 1),
    true
  );
  (*proportion_ones) = (*proportion_ones) / (float)this->_nrow;

  return proportion_ones;
}

void DesignMatrix::reduce_basis(float epsilon) {
  assert(epsilon >= 0);

  auto proportion_ones = this->proportion_ones();

  #pragma omp parallel for
  for (int c = 0; c < proportion_ones->shape().rows; c++) {
    if ((*proportion_ones)(c, 0) == 0.0f) {
      (*proportion_ones)(c, 0) = 1.1;
    }
  }

  float lower_bound = nc::min(*proportion_ones)(0, 0) * (1 + epsilon);
  auto shouldRemoved = nc::logical_or((*proportion_ones) >= 1.0f,
                                      (*proportion_ones) < lower_bound);

  #pragma omp parallel for
  for (int c = 0; c < this->_ncol; c++) {
    if (shouldRemoved(c, 0)) {
      ColIndices[c]._to_be_removed = true;
    }
  }

  auto& ColIndices = this->ColIndices;
  ColIndices.erase(std::remove_if(ColIndices.begin(), ColIndices.end(),
                                  [&](const ColIndex& col) mutable {
                                    return col._to_be_removed;
                                  }),
                   ColIndices.end());

  this->_ncol = ColIndices.size();

  cudaFree(this->_interaction_cuda);
  cudaFree(this->_len_interact_cuda);
  cudaFree(this->_sample_idx_cuda);
  
  this->_allocate_cudamem(true);
}


std::unique_ptr<nc::NdArray<bool>>
DesignMatrix::getCol(size_t col_idx, size_t start_idx, size_t end_idx) const {

  if (this->_type != "train") {
    std::cerr << "Should be called from DesignMatrix with type `train`"
              << std::endl;
  }
  const nc::NdArray<float> &df = this->_dataframe;
  auto& col_index = this->ColIndices[col_idx];

  // Check validality of col_index, return nullptr if invalid
  size_t interact_size = col_index.interaction.size();
  bool valid_sample_idx = (col_index.sample_idx < df.shape().rows);
  bool valid_interaction_size = (interact_size <= this->_max_order);
  bool increased = true;
  for (int i = 1; i < interact_size; i++) {
    increased &= (col_index.interaction[i - 1] < col_index.interaction[i]);
  }
  if (!valid_sample_idx | !valid_interaction_size | !increased) {
    return nullptr;
  }

  if ((start_idx == 0) & (end_idx == 0)) {
    end_idx = this->_nrow;
  }

  auto res = std::make_unique<nc::NdArray<bool>>(
      nc::ones<bool>(end_idx - start_idx, 1));
  for (auto c : col_index.interaction) {
    float thres = df(col_index.sample_idx, c);
    auto cur_col = df(nc::Slice(start_idx, end_idx), c);
    *res = nc::logical_and(*res, cur_col >= thres);
  }

  return res;
}

std::unique_ptr<BinSpMat> DesignMatrix::getRegion(uint64_t row_start,
                                                  uint64_t row_end,
                                                  uint64_t col_start,
                                                  uint64_t col_end) const {

  bool valid_row_bounds = (row_end <= this->_nrow) && (row_start <= row_end);
  bool valid_col_bounds = (col_end <= this->_ncol) && (col_start <= col_end);

  if (!valid_row_bounds || !valid_col_bounds) {
    throw std::out_of_range("Index out of range");
  }

  auto shifted_row_start = row_start + this->_offset;
  auto shifted_row_end = row_end + this->_offset;

  const size_t num_threads = omp_get_max_threads();
  const size_t row_size = shifted_row_end - shifted_row_start;
  const size_t col_size = col_end - col_start;
  size_t block_size = std::ceil(row_size / (float)num_threads);

  auto res = std::make_unique<BinSpMat>(row_size, col_size);

#pragma omp parallel
  {
    size_t thread_id = omp_get_thread_num();
    size_t block_row_start = shifted_row_start + thread_id * block_size;
    size_t block_row_end =
        shifted_row_start + std::min((thread_id + 1) * block_size, row_size);

    for (int row_idx = block_row_start; row_idx < block_row_end; row_idx++) {
      for (int col_idx = col_start; col_idx < col_end; col_idx++) {
        if (this->at(row_idx, col_idx)) {
          auto local_row_idx = row_idx - shifted_row_start;
          auto local_col_idx = col_idx - col_start;
          res->fill(local_row_idx, local_col_idx);
        }
      }
    }
  }

  res->translate();

  return res;
}

std::unique_ptr<BinSpMat> DesignMatrix::getBatch(const size_t start_idx,
                                                 const size_t end_idx) const {
  return getRegion(start_idx, end_idx, 0, this->_ncol);
}

bool DesignMatrix::at(const size_t row_idx, const size_t col_idx) const {
  auto &df = this->_dataframe;
  auto &col_index = this->ColIndices[col_idx];
  auto &interaction = col_index.interaction;
  auto sample_idx = col_index.sample_idx;

  bool res = true;
  for (auto &c : interaction) {
    float thres = df(sample_idx, c);
    float val = df(row_idx, c);
    res &= (val >= thres);
  }
  return res;
}
