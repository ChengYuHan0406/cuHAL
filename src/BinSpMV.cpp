#include "BinSpMV.hpp"
#include "omp.h"

/*
 *  This function is designed for parallel processing across rows.
 *  Caller must ensure that no two threads operate on the same row simultaneously.
 */
void BinSpMat::fill(const size_t row_idx, const size_t col_idx) {
  this->_row_info[row_idx + 1]++; 
  this->_col_info[row_idx].push_back(col_idx);
}

void BinSpMat::translate() {
  if (this->_translated) {
    return;
  }

  size_t nrow = this->_row_info.size() - 1;  

  /*
   * Convert `_row_info` to its prefix sum
   * and transfer the ownership to `_row_ptrs`
   */
  for (int r = 1; r <= nrow; r++) {
    this->_row_info[r] += this->_row_info[r - 1];
  }
  this->row_ptrs = std::move(this->_row_info);

  /* Number of non-zero elements */
  this->nnz = this->row_ptrs[nrow];

  /*
  * Flattens the 2-D vector `_col_info` into a 1-D vector `_col_indices`,
  * concatenating all its elements into a single contiguous sequence.
  */
  this->col_indices.resize(this->nnz);

  #pragma omp parallel for
  for (int r = 0; r < nrow; r++) {
    auto& cur_col_indices = this->_col_info[r]; 
    std::copy(
      cur_col_indices.begin(),
      cur_col_indices.end(),
      this->col_indices.begin() + this->row_ptrs[r]
    );
  }

  this->_translated = true;
}
