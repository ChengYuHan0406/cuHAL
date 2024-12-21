#pragma once
#include <memory>
#include <vector>
#include <NumCpp.hpp>

class BinSpMat {
public:
  BinSpMat(const size_t nrow, const size_t ncol) : _nrow(nrow),
                                                   _ncol(ncol),
                                                   _row_info(std::vector<size_t>(nrow + 1, 0)),
                                                   _col_info(nrow, std::vector<size_t>()),
                                                   _translated(false) {};


  BinSpMat(const BinSpMat& other) = default;
  BinSpMat(BinSpMat&& other) = default;

  void fill(const size_t row_idx, const size_t col_idx);
  void translate();
  std::unique_ptr<nc::NdArray<bool>> full() const;
  size_t nrow() const { return this->_nrow; }
  size_t ncol() const { return this->_ncol; }

  /* CSR format */
  std::vector<size_t> row_ptrs;
  std::vector<size_t> col_indices;

  /* Number of non-zero elements */
  size_t nnz;

private:
  size_t _nrow;
  size_t _ncol;
/*
 * Tracks the information needed for the `translate` operation
 * during the execution of the `fill` function.
 */
  std::vector<size_t> _row_info; // Record the number of non-zero elements in the previous row
  std::vector<std::vector<size_t>> _col_info; // Store the column indices of non-zero elements in the current row

  /* Indicates whether the current object has been translated */
  bool _translated;
};
