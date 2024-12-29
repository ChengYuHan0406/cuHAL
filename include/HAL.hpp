#pragma once
#include "BatchedDesignMatrix.hpp"
#include "DesignMatrix.hpp"
#include <NumCpp.hpp>
#include <memory>
#include "Loss.hpp"

class HAL {
public:
  HAL(const nc::NdArray<float>& dataframe, nc::NdArray<float> labels, size_t max_order, float sample_ratio = 1);
  const DesignMatrix& design_matrix() const { return this->_design_matrix; }
  const nc::NdArray<float>& labels() const { return this->_labels; }
  void update_weights(const size_t idx, const float delta);
  const nc::NdArray<float>& weights() const { return this->_weights; }
  float bias() const { return this->_bias; }
private:
  const nc::NdArray<float> _labels; // Denoted by 'y'
  const DesignMatrix _design_matrix; // Denoted by 'A'
  size_t _max_order;
  size_t _sample_ratio;
  nc::NdArray<float> _weights; // Denoted by 'beta'
  float _bias; // Denoted by 'b'
};

/* 
 * Parallel Stochastic Coordinate Descent
 *
 * Reference: 
 *  Joseph K. Bradley, et al. Parallel Coordinate Descent for L1 -Regularized Loss Minimization. ICML.
 */
class PSCDTrainer {
public:
  PSCDTrainer(HAL& hal,
              const Loss& loss,
              const float lambda,
              const float step_size) : _hal(hal), 
                                       _loss(loss),
                                       _lambda(lambda),
                                       _step_size(step_size),
                                       _label(hal.labels()) {} 
  void run_one_iteration();
private:
  HAL& _hal;
  const Loss& _loss;
  const float _lambda;
  const float _step_size;
  const nc::NdArray<float>& _label;
};

/*
class GDTrainer {
public:
  GDTrainer(HAL& hal,
            const Loss& loss,
            const float lambda,
            const float step_size) : _hal(hal), 
                                      _loss(loss),
                                      _lambda(lambda),
                                      _step_size(step_size),
                                      _label(hal.labels()) {} 
  void run(size_t batch_start, size_t batch_end);
private:
  HAL& _hal;
  const Loss& _loss;
  const float _lambda;
  const float _step_size;
  const nc::NdArray<float>& _label;
};
*/
class Predictor {
public:
  Predictor(const HAL& hal) : _hal(hal) {}
  std::unique_ptr<nc::NdArray<float>> predict(const nc::NdArray<float>& new_data) const;
private:
  const HAL& _hal;
};
