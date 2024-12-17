#include "DesignMatrix.hpp"
#include "NumCpp.hpp"
#include "omp.h"
#include <cfloat>
#include <cmath>
#include <memory>

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
    : _dataframe(dataframe), _max_order(max_order) {
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

std::unique_ptr<nc::NdArray<bool>>
DesignMatrix::getCol(const struct ColIndex &col_index, size_t start_idx,
                     size_t end_idx) const {
  const nc::NdArray<float> &df = this->_dataframe;

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

std::unique_ptr<nc::NdArray<bool>>
DesignMatrix::getBatch(const size_t start_idx, const size_t end_idx) const {
  const nc::NdArray<float> &df = this->_dataframe;
  size_t batch_size = end_idx - start_idx;
  size_t nrow = this->_nrow;

  bool valid_idx = (0 <= start_idx) & (start_idx < end_idx) & (end_idx <= nrow);
  if (!valid_idx) {
    return nullptr;
  }

  int num_threads = std::min(omp_get_max_threads(), static_cast<int>(this->_ncol));
  std::vector<nc::NdArray<bool>> vec_partial_res(num_threads);

  #pragma omp parallel num_threads(num_threads)
  {
    size_t ncol = this->_ncol;
    size_t block_size = std::floor(ncol / (num_threads - 1));

    int thread_id = omp_get_thread_num();

    int col_start = thread_id * block_size;
    int col_end = std::min((thread_id + 1) * block_size, ncol);

    std::vector<nc::NdArray<bool>> partial_res;

    for (int c = col_start; c < col_end; c++) {
      auto cur_col = this->getCol(this->ColIndices[c], start_idx, end_idx);
      partial_res.push_back(std::move(*cur_col));
    }

    vec_partial_res[thread_id] = nc::stack(partial_res, nc::Axis::COL);
  }

  auto res = std::make_unique<nc::NdArray<bool>>(nc::stack(vec_partial_res, nc::Axis::COL));
  return res;
}
