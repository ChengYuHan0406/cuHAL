#pragma once
#include <NumCpp.hpp>
#include "BinSpMV.hpp"
#include <memory>
#include <vector>

struct ColIndex {
  const std::vector<size_t> interaction;
  const size_t sample_idx;
}; 

class DesignMatrix {
public:
  DesignMatrix(const nc::NdArray<float>& dataframe, size_t max_order, float sample_ratio = 1);
  DesignMatrix(const DesignMatrix& other) = default;
  DesignMatrix(DesignMatrix&& other) = default;

  std::unique_ptr<nc::NdArray<bool>> getRegion(size_t row_start,
                                               size_t row_end,
                                               size_t col_start,
                                               size_t col_end) const;
  std::unique_ptr<nc::NdArray<bool>> getCol(size_t col_idx,
                                            size_t start_idx = 0,
                                            size_t end_idx = 0) const;
  std::unique_ptr<BinSpMat> getBatch(const size_t start_idx, const size_t end_idx) const;

  size_t get_nrow() const { return this->_nrow; }
  size_t get_ncol() const { return this->_ncol; }

  std::vector<struct ColIndex> ColIndices;

  /* For prediction */
  std::unique_ptr<DesignMatrix> getPredDesignMatrix(const nc::NdArray<float>& new_df) const;

private:
  nc::NdArray<float> _dataframe;
  std::string _type;
  size_t _max_order;
  size_t _nrow;
  size_t _ncol;
  size_t _offset;
  nc::NdArray<int> _sampled_row;
  void _init_ColIndices(size_t order, size_t prev_idx, std::vector<size_t>& interact);

  /* For prediction */
  void _init_PredDesignMatrix(const nc::NdArray<float>& new_df);
};
