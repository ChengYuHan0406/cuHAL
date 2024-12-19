#pragma once
#include <vector>

class BinSpMat {
public:
  BinSpMat(const size_t nrow) : _row_info(std::vector<size_t>(nrow + 1, 0)),
                                _col_info(nrow, std::vector<size_t>()),
                                _translated(false) {};
  void fill(const size_t row_idx, const size_t col_idx);
  void translate();

  /* CSR format */
  std::vector<size_t> row_ptrs;
  std::vector<size_t> col_indices;

  /* Number of non-zero elements */
  size_t nnz;

private:
/*
 * Tracks the information needed for the `translate` operation
 * during the execution of the `fill` function.
 */
  std::vector<size_t> _row_info; // Record the number of non-zero elements in the previous row
  std::vector<std::vector<size_t>> _col_info; // Store the column indices of non-zero elements in the current row

  /* Indicates whether the current object has been translated */
  bool _translated;
};
