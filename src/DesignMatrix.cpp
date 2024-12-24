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

std::unique_ptr<BinSpMat> 
DesignMatrix::getRegion(uint64_t row_start, uint64_t row_end,
                        uint64_t col_start, uint64_t col_end) const {

  bool valid_row_bounds = (row_end <= this->_nrow)
                       && (row_start <= row_end);
  bool valid_col_bounds = (col_end <= this->_ncol)
                       && (col_start <= col_end);

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
    size_t block_row_end = shifted_row_start + std::min((thread_id + 1) * block_size, row_size);

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

std::unique_ptr<BinSpMat> DesignMatrix::getBatch(const size_t start_idx, const size_t end_idx) const {
  return getRegion(start_idx, end_idx, 0, this->_ncol);
}

bool DesignMatrix::at(const size_t row_idx, const size_t col_idx) const {
  auto& df = this->_dataframe;
  auto& col_index = this->ColIndices[col_idx];
  auto& interaction = col_index.interaction;
  auto sample_idx = col_index.sample_idx;

  bool res = true; 
  for (auto& c : interaction) {
    float thres = df(sample_idx, c);
    float val = df(row_idx, c);
    res &= (val >= thres);
  }
  return res;
}

