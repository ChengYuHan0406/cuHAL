#pragma once
#include "NumCpp.hpp"
#include <memory>
#include <vector>

struct ColIndex {
  const std::vector<size_t> interaction;
  const size_t sample_idx;
}; 

class DesignMatrix {
public:
  DesignMatrix(const nc::NdArray<double>& dataframe, size_t max_order);
  std::unique_ptr<nc::NdArray<bool>> getCol(const struct ColIndex& col_index,
                                            size_t start_idx = 0,
                                            size_t end_idx = 0) const;
  std::unique_ptr<nc::NdArray<bool>> getBatch(const size_t start_idx, const size_t end_idx);

  std::vector<struct ColIndex> ColIndices;
private:
  const nc::NdArray<double> _dataframe;
  size_t _max_order;
  size_t _nrow;
  size_t _ncol;
  void _init_ColIndices(size_t order, size_t prev_idx, std::vector<size_t>& interact);
};
