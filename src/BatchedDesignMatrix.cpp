#include "BatchedDesignMatrix.hpp"
#include "DesignMatrix.hpp"
#include <NumCpp.hpp>

std::unique_ptr<nc::NdArray<bool>> BatchedDesignMatrix::get(size_t index) const {
  auto nrow = this->_design_matrix.get_nrow();
  auto start_idx = index * this->_batch_size;
  auto end_idx = std::min((index + 1) * this->_batch_size, nrow);

  if (start_idx >= nrow) {
    throw std::out_of_range("Index out of range");
  }
  return this->_design_matrix.getBatch(start_idx, end_idx);
} 

size_t BatchedDesignMatrix::size() const {
  auto nrow = this->_design_matrix.get_nrow();
  return std::floor((nrow + this->_batch_size - 1) / this->_batch_size);
} 
