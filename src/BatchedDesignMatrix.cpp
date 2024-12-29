#include "BatchedDesignMatrix.hpp"
#include "BinSpMV.hpp"
#include "DesignMatrix.hpp"
#include <NumCpp.hpp>
#include <memory>

std::unique_ptr<BinSpMat> BatchedDesignMatrix::get(size_t index) {
  auto nrow = this->_design_matrix.get_nrow();
  auto start_idx = index * this->_batch_size;
  auto end_idx = std::min((index + 1) * this->_batch_size, nrow);

  if (start_idx >= nrow) {
    throw std::out_of_range("Index out of range");
  }

  if (index == this->_prefetched_idx) {
    auto res = std::make_unique<BinSpMat>(std::move(*this->_prefetched_data));
    this->_prefetched_idx = -1;
    return res;
  }

  return this->_design_matrix.getBatch(start_idx, end_idx);
} 


std::unique_ptr<nc::NdArray<float>> BatchedDesignMatrix::batchedMV(
  size_t index,
  const nc::NdArray<float>& vec,
  bool transpose
) const {
  
  auto nrow = this->_design_matrix.get_nrow();
  auto start_idx = index * this->_batch_size;
  auto end_idx = std::min((index + 1) * this->_batch_size, nrow);

  if (start_idx >= nrow) {
    throw std::out_of_range("Index out of range");
  }

  return this->_design_matrix.fusedRegionMV(
    start_idx,
    end_idx,
    0,
    this->_design_matrix.get_ncol(),
    vec,
    transpose
  );
}

void BatchedDesignMatrix::prefetch(size_t index) {
  this->_prefetched_data = this->get(index);
  this->_prefetched_idx = index;
}

size_t BatchedDesignMatrix::len() const {
  auto nrow = this->_design_matrix.get_nrow();
  return std::floor((nrow + this->_batch_size - 1) / this->_batch_size);
} 
