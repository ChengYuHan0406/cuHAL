#pragma once
#include "BinSpMV.hpp"
#include <NumCpp.hpp>
#include <cstddef>
#include <memory>
#include <unordered_set>
#include <vector>

struct ColIndex {
  std::vector<size_t> interaction;
  size_t sample_idx;

  ColIndex(const ColIndex&) = default;
  ColIndex& operator=(const ColIndex&) = default;
  ColIndex(ColIndex&& other) = default;
  ColIndex& operator=(ColIndex&& other) = default;

  bool _to_be_removed = false;
};

class DesignMatrix {
public:
  DesignMatrix(const nc::NdArray<float> &dataframe, size_t max_order,
               float sample_ratio = 1, float reduce_epsilon = -1);
  DesignMatrix(const DesignMatrix &other);
  DesignMatrix(DesignMatrix &&other) = delete;
  ~DesignMatrix();

  std::unique_ptr<BinSpMat> getRegion(size_t row_start, size_t row_end,
                                      size_t col_start, size_t col_end) const;
  std::unique_ptr<nc::NdArray<bool>>
  getCol(size_t col_idx, size_t start_idx = 0, size_t end_idx = 0) const;
  std::unique_ptr<BinSpMat> getBatch(const size_t start_idx,
                                     const size_t end_idx) const;
  std::unique_ptr<nc::NdArray<float>>
  fusedRegionMV(size_t row_start, size_t row_end, size_t col_start,
                size_t col_end, const nc::NdArray<float> &x,
                bool transpose = false,
                const nc::NdArray<int>& col_perm = {-1}) const;
  std::unique_ptr<nc::NdArray<float>>
  fusedColSubsetMV(const nc::NdArray<float>& x,
                   const nc::NdArray<int>& colidx_subset,
                   bool transpose = false) const;

  size_t get_nrow() const { return this->_nrow; }
  size_t get_ncol() const { return this->_ncol; }

  bool at(const size_t row_idx, const size_t col_idx) const;

  std::vector<struct ColIndex> ColIndices;

  std::unique_ptr<nc::NdArray<float>> proportion_ones() const; 
  void reduce_basis(float epsilon);

  /* For prediction */
  std::unique_ptr<DesignMatrix>
  getPredDesignMatrix(const nc::NdArray<float> &new_df) const;


private:
  nc::NdArray<float> _dataframe;
  std::string _type;
  size_t _max_order;
  size_t _nrow;
  size_t _ncol;
  size_t _offset;
  nc::NdArray<int> _sampled_row;
  void _init_ColIndices(size_t order, int prev_idx,
                        std::vector<size_t> &interact);

  /* For prediction */
  void _init_PredDesignMatrix(const nc::NdArray<float> &new_df);

  /* Cuda */
  void _allocate_cudamem(bool reserve_df = false);
  float *_dataframe_cuda;
  size_t *_interaction_cuda;
  size_t *_len_interact_cuda;
  size_t *_sample_idx_cuda;
};

/* The `VectorHash` implementation is borrowed from 
 * https://stackoverflow.com/questions/29855908/c-unordered-set-of-vectors */
struct VectorHash {
    size_t operator()(const std::vector<float>& v) const {
        std::hash<float> hasher;
        size_t seed = 0;
        for (int i : v) {
            seed ^= hasher(i) + 0x9e3779b9 + (seed<<6) + (seed>>2);
        }
        return seed;
    }
};

using CutPoints = std::unordered_set<std::vector<float>, VectorHash>;
