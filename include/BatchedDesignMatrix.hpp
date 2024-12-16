#include "DesignMatrix.hpp"
#include "NumCpp.hpp"
#include <cmath>
#include <memory>

class BatchedDesignMatrix {
public:
  BatchedDesignMatrix(const DesignMatrix& design_matrix,
                      size_t batch_size) : _design_matrix(design_matrix),
                                           _batch_size(batch_size) {}
  std::unique_ptr<nc::NdArray<bool>> get(size_t index) const; 
  size_t size() const;

private:
  const DesignMatrix& _design_matrix;
  const size_t _batch_size;
};
