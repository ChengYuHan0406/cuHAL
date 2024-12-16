#include "DesignMatrix.hpp"
#include "NumCpp.hpp"
#include "omp.h"
#include <cfloat>
#include <cmath>
#include <memory>

size_t static _comb(size_t n, size_t k);
size_t static _design_matrix_col_size(size_t nrow, size_t ncol,
                                      size_t max_order);

void DesignMatrix::_init_ColIndices(size_t order, size_t prev_idx,
                                    std::vector<size_t> &interact) {
  if (order == 0) {
    for (int i = 0; i < this->_nrow; i++) {
      this->ColIndices.push_back(ColIndex{interact, static_cast<size_t>(i)});
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

DesignMatrix::DesignMatrix(const nc::NdArray<double> &dataframe,
                           size_t max_order)
    : _dataframe(dataframe), _max_order(max_order) {
  auto df_shape = dataframe.shape();
  size_t df_nrow = df_shape.rows;
  size_t df_ncol = df_shape.cols;
  this->_nrow = df_nrow;
  this->_ncol = _design_matrix_col_size(df_nrow, df_ncol, max_order);

  for (int o = 1; o <= max_order; o++) {
    std::vector<size_t> interact;
    this->_init_ColIndices(o, -1, interact);
  }
};

std::unique_ptr<nc::NdArray<bool>>
DesignMatrix::getCol(const struct ColIndex &col_index, size_t start_idx,
                     size_t end_idx) const {
  const nc::NdArray<double> &df = this->_dataframe;

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
    double thres = df(col_index.sample_idx, c);
    auto cur_col = df(nc::Slice(start_idx, end_idx), c);
    *res = nc::logical_and(*res, cur_col >= thres);
  }

  return res;
}

std::unique_ptr<nc::NdArray<bool>>
DesignMatrix::getBatch(const size_t start_idx, const size_t end_idx) {
  const nc::NdArray<double> &df = this->_dataframe;
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

/********************************* Helper Functions ***************************************/

size_t static _comb(size_t n, size_t k) {
  size_t numerator = 1;
  size_t denominator = 1;

  while (k > 0) {
    numerator *= n;
    denominator *= k;
    n--, k--;
  }

  return numerator / denominator;
}

size_t static _design_matrix_col_size(size_t nrow, size_t ncol,
                                      size_t max_order) {
  size_t col_size = 0;
  for (int i = 1; i <= max_order; i++) {
    col_size += _comb(ncol, i);
  }
  col_size *= nrow;

  return col_size;
}
