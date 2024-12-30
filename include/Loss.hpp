#pragma once
#include <NumCpp.hpp>
#include <NumCpp/Functions/empty.hpp>
#include <cassert>

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
