#include "DesignMatrix.hpp"
#include <NumCpp.hpp>
#include "omp.h"
#include <NumCpp/Core/Enums.hpp>
#include <NumCpp/Functions/stack.hpp>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sys/types.h>

void DesignMatrix::_init_ColIndices(size_t order, size_t prev_idx,
                                    std::vector<size_t> &interact) {
  if (order == 0) {
    auto num_sampled_row = this->_sampled_row.shape().cols;
    for (int i = 0; i < num_sampled_row; i++) {
      int row_idx = this->_sampled_row(0, i);
      this->ColIndices.push_back(ColIndex{interact, static_cast<size_t>(row_idx)});
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
                           size_t max_order, float sample_ratio)
    : _dataframe(dataframe), _max_order(max_order), 
      _type("train"), _offset(0) {
  auto df_shape = dataframe.shape();
  size_t df_nrow = df_shape.rows;
  size_t df_ncol = df_shape.cols;
  this->_nrow = df_nrow;

  if (sample_ratio == 1) {
    this->_sampled_row = nc::arange<int>(0, this->_nrow).reshape(1, this->_nrow);
  } else {
    uint32_t num_sampled_row = std::floor(this->_nrow * sample_ratio);
    this->_sampled_row = nc::random::randInt<int>({1, num_sampled_row}, 0, this->_nrow);
  }
  for (int o = 1; o <= max_order; o++) {
    std::vector<size_t> interact;
    this->_init_ColIndices(o, -1, interact);
  }

  this->_ncol = this->ColIndices.size();  
};

void DesignMatrix::_init_PredDesignMatrix(const nc::NdArray<float>& new_df) {
  this->_offset = this->_dataframe.shape().rows; 
  this->_dataframe = nc::stack({this->_dataframe, new_df}, nc::Axis::ROW);
  this->_type = "prediction";
  this->_nrow = new_df.shape().rows;
}

std::unique_ptr<DesignMatrix> DesignMatrix::getPredDesignMatrix(const nc::NdArray<float>& new_df) const {
  if (this->_type != "train") {
    std::cerr << "Should be called from DesignMatrix with type `train`" << std::endl;
  }
  auto res = std::make_unique<DesignMatrix>(*this);
  res->_init_PredDesignMatrix(new_df);
  return res;
}

std::unique_ptr<nc::NdArray<bool>> 
DesignMatrix::getRegion(uint64_t row_start, uint64_t row_end,
                        uint64_t col_start, uint64_t col_end) const {
  
  if (row_end > this->_nrow || col_end > this->_ncol) {
    throw std::out_of_range("Index out of range");
  }

  auto& df = this->_dataframe;
  const uint64_t num_threads = std::min(omp_get_max_threads(), static_cast<int>(this->_ncol));
  const uint64_t row_size = row_end - row_start;
  const uint64_t col_size = col_end - col_start;
  const uint64_t num_elements = row_size * col_size; 
  uint64_t block_size = std::ceil(num_elements / (float)num_threads);

  /* Since NumCpp uses uint32_t for array size, 
   * ensure that `row_size * col_size` does not exceed the uint32_t limit. */
  if (row_size * col_size >= std::numeric_limits<uint32_t>::max()) {
    throw std::out_of_range("Size exceeds uint32_t limit");
  }

  auto res = std::make_unique<nc::NdArray<bool>>(row_size, col_size);

  #pragma omp parallel num_threads(num_threads)
  {
    uint64_t thread_id = omp_get_thread_num();
    uint64_t idx_start = thread_id * block_size;
    uint64_t idx_end = std::min((thread_id + 1) * block_size, num_elements);

    for (int64_t i = idx_start; i < (int64_t)idx_end; i++) {
      uint64_t local_row_idx = i / col_size;
      uint64_t local_col_idx = i % col_size;
      auto global_row_idx = row_start + local_row_idx;
      auto global_col_idx = col_start + local_col_idx; 
      auto& col_index = this->ColIndices[global_col_idx]; 
      auto& interaction = col_index.interaction;
      auto sample_idx = col_index.sample_idx;

      bool cur_elem = true; 
      for (auto& c : interaction) {
        float thres = df(sample_idx, c);
        float val = df(global_row_idx, c);
        cur_elem &= (val >= thres);
      }

      (*res)(local_row_idx, local_col_idx) = cur_elem; 
    }
  }

  return res;
}

std::unique_ptr<nc::NdArray<bool>>
DesignMatrix::getCol(size_t col_idx, size_t start_idx,
                     size_t end_idx) const {

  if (this->_type != "train") {
    std::cerr << "Should be called from DesignMatrix with type `train`" << std::endl;
  }
  const nc::NdArray<float> &df = this->_dataframe;
  auto col_index = this->ColIndices[col_idx];

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

std::unique_ptr<BinSpMat> DesignMatrix::getBatch(const size_t start_idx, const size_t end_idx) const {

  auto shifted_start_idx = start_idx + this->_offset;
  auto shifted_end_idx = end_idx + this->_offset;
  
  if (shifted_end_idx > this->_dataframe.shape().rows || shifted_start_idx > shifted_end_idx) {
    throw std::out_of_range("Index out of range");
  }

  auto& df = this->_dataframe;

  const size_t num_threads = omp_get_max_threads();
  const size_t batch_size = shifted_end_idx - shifted_start_idx;
  size_t block_size = std::ceil(batch_size / (float)num_threads);

  auto res = std::make_unique<BinSpMat>(batch_size, this->_ncol);

  #pragma omp parallel
  {
    size_t thread_id = omp_get_thread_num();
    size_t row_start = shifted_start_idx + thread_id * block_size;
    size_t row_end = shifted_start_idx + std::min((thread_id + 1) * block_size, batch_size);

    for (int row_idx = row_start; row_idx < row_end; row_idx++) {
      for (int col_idx = 0; col_idx < this->_ncol; col_idx++) {
        auto& col_index = this->ColIndices[col_idx];
        auto& interaction = col_index.interaction;
        auto sample_idx = col_index.sample_idx;

        bool cur_elem = true; 
        for (auto& c : interaction) {
          float thres = df(sample_idx, c);
          float val = df(row_idx, c);
          cur_elem &= (val >= thres);
        }

        if (cur_elem) {
          auto local_row_idx = row_idx - shifted_start_idx;
          res->fill(local_row_idx, col_idx);
        }
      }
    }
  }

  res->translate();

  return res;
}


