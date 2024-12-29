#pragma once
#include "BatchedDesignMatrix.hpp"
#include "DesignMatrix.hpp"
#include <NumCpp.hpp>
#include <NumCpp/Functions/zeros.hpp>
#include <memory>
#include "Loss.hpp"

class HAL {
public:
  HAL(const nc::NdArray<float>& dataframe, nc::NdArray<float> labels, size_t max_order, float sample_ratio = 1);
  const DesignMatrix& design_matrix() const { return this->_design_matrix; }
  const nc::NdArray<float>& labels() const { return this->_labels; }
  void update_weights(const size_t idx, const float delta);
  void set_weights(const nc::NdArray<float>& new_weights);
  void set_bias(const float new_bias);
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

class AdamTrainer {
public:
  AdamTrainer(HAL& hal,
            const size_t batch_size,
            const Loss& loss,
            const float lambda,
            const float step_size,
            const float beta_1 = 0.9,
            const float beta_2 = 0.999) : _hal(hal), 
                                          _batch_size(batch_size),
                                          _bdm(hal.design_matrix(), batch_size),
                                          _loss(loss),
                                          _lambda(lambda),
                                          _step_size(step_size),
                                          _beta_1(beta_1),
                                          _beta_2(beta_2),
                                          _u_weights(nc::zeros<float>({hal.weights().shape().rows, 1})),
                                          _v_weights(nc::zeros<float>({hal.weights().shape().rows, 1})),
                                          _u_bias(0),
                                          _v_bias(0),
                                          _num_iter(0),
                                          _label(hal.labels()) {} 
  void run(size_t batch_idx);
  size_t len() { return this->_bdm.len(); }
private:
  HAL& _hal;
  const size_t _batch_size;
  BatchedDesignMatrix _bdm;
  const Loss& _loss;
  const float _lambda;
  const float _step_size;
  const float _beta_1; 
  const float _beta_2; 
  nc::NdArray<float> _u_weights;
  nc::NdArray<float> _v_weights;
  float _u_bias;
  float _v_bias;
  size_t _num_iter;
  const nc::NdArray<float>& _label;
  size_t _batched_start(size_t batch_idx) const;
  size_t _batched_end(size_t batch_idx) const;
};

class Predictor {
public:
  Predictor(const HAL& hal) : _hal(hal) {}
  std::unique_ptr<nc::NdArray<float>> predict(const nc::NdArray<float>& new_data) const;
private:
  const HAL& _hal;
};
