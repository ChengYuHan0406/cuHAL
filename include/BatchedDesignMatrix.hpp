#include "BinSpMV.hpp"
#include "DesignMatrix.hpp"
#include <NumCpp.hpp>
#include <cmath>
#include <memory>

class BatchedDesignMatrix {
public:
  BatchedDesignMatrix(const DesignMatrix& design_matrix,
                      size_t batch_size) : _design_matrix(design_matrix),
                                           _batch_size(batch_size),
                                           _prefetched_idx(-1) {}

  BatchedDesignMatrix(const BatchedDesignMatrix& other) = delete;
  BatchedDesignMatrix(BatchedDesignMatrix&& other) = delete;

  std::unique_ptr<BinSpMat> get(size_t index); 
  void prefetch(size_t index);
  size_t len() const;
  size_t batch_size() const { return _batch_size; }
  size_t nrow() const { return this->_design_matrix.get_nrow(); }
  size_t ncol() const { return this->_design_matrix.get_ncol(); }

private:
  const DesignMatrix& _design_matrix;
  const size_t _batch_size;
  std::unique_ptr<BinSpMat> _prefetched_data;
  int _prefetched_idx;
};
