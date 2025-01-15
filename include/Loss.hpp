#pragma once
#include <NumCpp.hpp>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <omp.h>

#define PROLOGUE \
    assert(output.shape() == label.shape());\
    assert(output.shape().cols == 1);\
    size_t len = output.shape().rows;

class Loss {
public:
  virtual float compute(const nc::NdArray<float>& output,
                        const nc::NdArray<float>& label,
                        const nc::NdArray<float>& weight = nc::empty<float>(0, 0)) const = 0;
  /* Gradient with respect to output */
  virtual nc::NdArray<float> grad(const nc::NdArray<float>& output,
                                  const nc::NdArray<float>& label,
                                  const nc::NdArray<float>& weight = nc::empty<float>(0, 0)) const = 0;
};

class MSELoss : public Loss {
public:
  float compute(const nc::NdArray<float>& output,
                const nc::NdArray<float>& label,
                const nc::NdArray<float>& weight = nc::empty<float>(0, 0)) const {
    PROLOGUE

    return nc::sum(nc::power((output - label), 2))(0, 0) / len;
  }

  nc::NdArray<float> grad(const nc::NdArray<float>& output,
                          const nc::NdArray<float>& label,
                          const nc::NdArray<float>& weight = nc::empty<float>(0, 0)) const {
    PROLOGUE
    auto res = (((2 / (float)len) * (output - label))).reshape(1, len);
    return res;
  }
};

class LogitLoss : public Loss {
public:
  float compute(const nc::NdArray<float>& output,
                const nc::NdArray<float>& label,
                const nc::NdArray<float>& weight = nc::empty<float>(0, 0)) const {
    PROLOGUE
    auto p = 1.0f / (1.0f + nc::exp(-output));
    auto res = -nc::mean(label * nc::log(p) + (1.0f - label) * nc::log(1.0f - p))(0, 0);
    return res;
  }

  nc::NdArray<float> grad(const nc::NdArray<float>& output,
                          const nc::NdArray<float>& label,
                          const nc::NdArray<float>& weight = nc::empty<float>(0, 0)) const {
    PROLOGUE
    auto p = 1.0f / (1.0f + nc::exp(-output));
    auto res = (p - label) / (float)len;
    return res.reshape(1, len);
  }
};

class WMAELoss : public Loss {
public:
  float compute(const nc::NdArray<float>& output,
                const nc::NdArray<float>& label,
                const nc::NdArray<float>& weight) const {
    PROLOGUE

    return nc::dot(nc::abs(output - label), weight)(0, 0) / nc::sum(weight)(0, 0);
  }

  nc::NdArray<float> grad(const nc::NdArray<float>& output,
                          const nc::NdArray<float>& label,
                          const nc::NdArray<float>& weight) const {
    PROLOGUE
    auto res = (weight * nc::sign(output - label).astype<float>()) / nc::sum(weight)(0, 0);
    return res.reshape(1, len);
  }
};

/* Assume `label` is a 2D array of shape N x 2,
 * where the first column represents T and the second column represents S. */
class CoxNLogPL : public Loss {
public:
  float compute(const nc::NdArray<float>& output,
                const nc::NdArray<float>& label,
                const nc::NdArray<float>& weight = nc::empty<float>(0, 0)) const {

    auto len = output.shape().rows;
    auto shifted_output = output - nc::max(output);

    this->_sort(label);
    this->_fill_suffix(shifted_output, label);

    auto T = label(label.rSlice(), 0);
    auto S = label(label.rSlice(), 1);

    float res = 0;
    for (int i = 0; i < len; i++) {
      if (S(i, 0) == 1) {
        res += (shifted_output(i, 0) - nc::log(this->_suffix[T(i, 0)]));
      }
    }
    return -(res / (float)len);
  }

  nc::NdArray<float> grad(const nc::NdArray<float>& output,
                          const nc::NdArray<float>& label,
                          const nc::NdArray<float>& weight = nc::empty<float>(0, 0)) const {
    uint32_t len = output.shape().rows;
    auto shifted_output = output - nc::mean(output).astype<float>();

    this->_sort(label);
    this->_fill_suffix(shifted_output, label);
    this->_fill_prefix(shifted_output, label);

    auto T = label(label.rSlice(), 0);
    auto S = label(label.rSlice(), 1);

    auto res = nc::NdArray<float>(1, len);

    #pragma omp parallel for
    for (int i = 0; i < len; i++) {
      res(0, i) = (S(i, 0) - nc::exp(shifted_output(i, 0)) * this->_prefix[T(i, 0)]);
    }

    return -(res / (float)len);
  }

private:
  void _sort(const nc::NdArray<float>& label) const {
    this->_sorted_idx = nc::argsort(label(label.rSlice(), 0),
                                    nc::Axis::ROW);
  }

  /* Assume `_sort` has been called */
  void _fill_suffix(const nc::NdArray<float>& output,
                    const nc::NdArray<float>& label) const {
    auto len = output.shape().rows;
    this->_suffix.clear();
    double cur_suffix = 0;
    auto T = label(label.rSlice(), 0);

    for (int i = len - 1; i >= 0; i--) {
      auto cur_idx = this->_sorted_idx(i, 0);
      cur_suffix += nc::exp(output(cur_idx, 0));
      this->_suffix[T(cur_idx, 0)] = cur_suffix;
    }
  }

  /* Assume `_sort` and `_fill_suffix` have been called */
  void _fill_prefix(const nc::NdArray<float>& output,
                    const nc::NdArray<float>& label) const {
    auto len = output.shape().rows;
    this->_prefix.clear();
    double cur_prefix = 0;
    auto T = label(label.rSlice(), 0);
    auto S = label(label.rSlice(), 1);

    for (int i = 0; i < len; i++) {
      auto cur_idx = this->_sorted_idx(i, 0);
      cur_prefix += (S(cur_idx, 0) / this->_suffix[T(cur_idx, 0)]);
      this->_prefix[T(cur_idx, 0)] = cur_prefix;
    }
  }

  /* Map "T_i" |-> "sum_{k : T_k >= T_i} [ exp( output[k] ) ]" */
  mutable std::unordered_map<float, double> _suffix;
  /* Map "T_i" |-> "sum_{j : T_j <= T_i} [ S_j / suffix[T_j] ]" */
  mutable std::unordered_map<float, double> _prefix;
  /* Increasing in T_i */
  mutable nc::NdArray<uint32_t> _sorted_idx;
};
