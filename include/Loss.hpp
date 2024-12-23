#pragma once
#include <NumCpp.hpp>
#include <cassert>

#define PROLOGUE \
    assert(output.shape() == label.shape());\
    assert(output.shape().cols == 1);\
    size_t len = output.shape().rows;

class Loss {
public:
  virtual float compute(const nc::NdArray<float>& output, const nc::NdArray<float>& label) const = 0;
  /* Gradient with respect to output */
  virtual nc::NdArray<float> grad(const nc::NdArray<float>& output, const nc::NdArray<float>& label) const = 0;
};

class MSELoss : public Loss {
public:
  float compute(const nc::NdArray<float>& output, const nc::NdArray<float>& label) const {
    PROLOGUE

    return nc::sum(nc::power((output - label), 2))(0, 0) / len;
  }

  nc::NdArray<float> grad(const nc::NdArray<float>& output,
                          const nc::NdArray<float>& label) const {
    PROLOGUE
    auto res = (((2 / (float)len) * (output - label))).reshape(1, len);
    return res;
  }
};
