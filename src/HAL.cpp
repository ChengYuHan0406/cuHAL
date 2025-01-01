#include "HAL.hpp"
#include "DesignMatrix.hpp"
#include <NumCpp.hpp>
#include "assert.h"
#include <NumCpp/Core/Slice.hpp>
#include <NumCpp/Functions/clip.hpp>
#include <NumCpp/Functions/empty.hpp>
#include <NumCpp/Functions/logical_or.hpp>
#include <NumCpp/Functions/maximum.hpp>
#include <NumCpp/Functions/mean.hpp>
#include <NumCpp/Functions/power.hpp>
#include <NumCpp/Functions/sqrt.hpp>
#include <NumCpp/Functions/square.hpp>
#include <NumCpp/Functions/sum.hpp>
#include <NumCpp/Functions/zeros.hpp>
#include <cmath>
#include "omp.h"
#include <cstdlib>
#include <stdlib.h> 
#include <time.h>

std::unique_ptr<nc::NdArray<float>> batch_binspmv(BatchedDesignMatrix& A, const nc::NdArray<float>& x);

HAL::HAL(const nc::NdArray<float>& dataframe,
         nc::NdArray<float> labels,
         size_t max_order,
         float sample_ratio,
         float reduce_epsilon) : _design_matrix(dataframe, max_order, sample_ratio, reduce_epsilon),
                               _labels(labels),
                               _max_order(max_order),
                               _sample_ratio(sample_ratio),
                               _bias(0) {

  auto shape_dataframe = dataframe.shape();
  auto shape_labels = labels.shape();

  /* Check data shape */
  assert(shape_labels.cols == 1);
  assert(shape_dataframe.rows == shape_labels.rows);

  auto nrow = this->_design_matrix.get_nrow();
  auto ncol = this->_design_matrix.get_ncol();
  
  /* Initialize `_weights` */
  this->_weights = nc::zeros<float>({static_cast<uint32_t>(ncol), 1}); 
}

void HAL::update_weights(const size_t idx, const float delta) {
  if (idx == 0) {
    this->_bias += delta;
  } else {
    this->_weights(idx - 1, 0) += delta;
  }
}

void HAL::set_weights(const nc::NdArray<float>& new_weights) {
  this->_weights = new_weights;
}
void HAL::set_bias(const float new_bias) {
  this->_bias = new_bias;
}

float _soft_threshold(float w, float lamb, float step_size) {
  auto thres = lamb * step_size;
  if (w > thres) {
    return w - thres;
  } else if (w < -thres) {
    return w + thres;
  } else {
    return 0;
  }
}

nc::NdArray<float>
_soft_threshold(const nc::NdArray<float>& w, float lamb, float step_size) {
  auto thres = lamb * step_size;
  auto right = (w > thres).astype<float>();
  auto left = (w < -thres).astype<float>();

  return (w - thres) * right + (w + thres) * left;
}

void PSCDTrainer::run_one_iteration() {
  auto outputs = this->_hal.design_matrix().fusedRegionMV(
    0, this->_hal.design_matrix().get_nrow(),
    0, this->_hal.design_matrix().get_ncol(),
    this->_hal.weights()
  );
  (*outputs) += this->_hal.bias();
  auto grad = this->_loss.grad(*outputs, this->_label); 

  size_t num_threads = omp_get_max_threads();
  std::vector<std::pair<size_t, float>> deltas(num_threads); // vector of (column idx, delta)

  #pragma omp parallel num_threads(num_threads)
  {
    size_t thread_idx = omp_get_thread_num();
    /* Number of columns including bias term */
    auto ncol = this->_hal.design_matrix().get_ncol() + 1;
    auto col_idx = rand() % ncol;

    float partial_grad;
    float weight;

    if (col_idx != 0) {
      auto cur_col = this->_hal.design_matrix().getCol(col_idx - 1);
      weight = this->_hal.weights()(col_idx - 1, 0);
      partial_grad = nc::dot(grad, cur_col->astype<float>())(0, 0);
    } else {
      weight = this->_hal.bias();
      partial_grad = nc::sum(grad)(0, 0);
    }

    float delta = _soft_threshold(weight - partial_grad * this->_step_size,
                                  this->_lambda,
                                  this->_step_size) - weight;
    deltas[thread_idx] = std::pair<size_t, float>(col_idx, delta);
  }

  for (auto p : deltas) {
    this->_hal.update_weights(p.first, p.second);
  }
}

nc::NdArray<float> SRTrainer::grad_wrt_outputs() {
  auto& design_matrix = this->_hal.design_matrix();
  auto& weight =  this->_hal.weights();

  auto outputs = design_matrix.fusedRegionMV(
    0, this->_hal.design_matrix().get_nrow(),
    0, this->_hal.design_matrix().get_ncol(),
    weight
  );
  (*outputs) += this->_hal.bias();
  auto grad = this->_loss.grad(*outputs, this->_label).transpose(); 

  return grad;
}

std::unique_ptr<nc::NdArray<float>> SRTrainer::partial_derivs() {
  auto& design_matrix = this->_hal.design_matrix();
  auto grad = this->grad_wrt_outputs();

  auto partial_derivs = design_matrix.fusedRegionMV(
    0, this->_hal.design_matrix().get_nrow(),
    0, this->_hal.design_matrix().get_ncol(),
    grad,
    true
  );

  return partial_derivs;
}

float SRTrainer::partial_deriv_bias() {
  return nc::sum(this->grad_wrt_outputs())(0, 0);
}

SRTrainer::SRTrainer(HAL& hal,
                     const Loss& loss,
                     const float step_size,
                     const size_t max_iters) : _hal(hal), 
                                              _loss(loss),
                                              _step_size(step_size),
                                              _max_iters(max_iters),
                                              _label(hal.labels()),
                                              _epsilon(1e-3),
                                              _num_lambdas(100) {
  
  auto& design_matrix = this->_hal.design_matrix();
  auto partial_grad = this->partial_derivs(); 
  this->_lambda_max = nc::max(nc::abs(*partial_grad))(0, 0);
  this->_lambda_min = this->_epsilon * this->_lambda_max;
  this->_lambda_step = std::pow(this->_epsilon,
                                -(1 / (float)(this->_num_lambdas - 1)));
} 

void SRTrainer::run(const nc::NdArray<float>& val_df,
                    const nc::NdArray<float>& val_label) {
  auto predictor = Predictor(this->_hal);

  float cur_lambda = this->_lambda_max;
  float prev_lambda = 0;

  for (int i = 0; i < this->_num_lambdas; i++) {
    std::cout << "Lambda = " << cur_lambda << std::endl;
    this->solve_lambda(cur_lambda, prev_lambda);

    auto val_out = *predictor.predict(val_df);
    auto val_loss = this->_loss.compute(val_out, val_label);
    std::cout << "Validation Loss: " << val_loss << std::endl;

    prev_lambda = cur_lambda;
    cur_lambda = prev_lambda / this->_lambda_step;
  }
}

void SRTrainer::solve_lambda(float cur_lambda, float prev_lambda) {
  auto& design_matrix = this->_hal.design_matrix();

  auto thres = 2 * cur_lambda - prev_lambda;
  auto nonzeros = (this->_hal.weights() != 0.0f);
  auto strong_set = (nc::abs(*this->partial_derivs()) > thres);
  strong_set = nc::logical_or(strong_set, nonzeros);
  bool include_bias = (this->_hal.bias() != 0.0f ||
                       this->partial_deriv_bias() > thres);

  bool KKT_holds = false;
  
  int iters = 0;
  while (!KKT_holds && iters++ < this->_max_iters) {
    /* Cyclic coordinate descent on strong set */
    if (include_bias) {
      float delta = _soft_threshold(this->_hal.bias() - this->partial_deriv_bias() * this->_step_size,
                                    cur_lambda,
                                    this->_step_size) - this->_hal.bias();

      this->_hal.update_weights(0, delta);
    }

    for (int c = 0; c < design_matrix.get_ncol(); c++) {
      if (!strong_set(c, 0)) {
        continue;
      }

      auto cur_col = design_matrix.getCol(c);
      auto weight = this->_hal.weights()(c, 0);
      auto partial_grad = nc::dot(this->grad_wrt_outputs(), cur_col->astype<float>())(0, 0);

      float delta = _soft_threshold(weight - partial_grad * this->_step_size,
                                    cur_lambda,
                                    this->_step_size) - weight;

      this->_hal.update_weights(c + 1, delta);
    }

    /* Check KKT */
    auto partial_deriv_weight = this->partial_derivs();
    auto partial_deriv_bias = this->partial_deriv_bias();

    auto cond_weight = (nc::abs(*partial_deriv_weight) > cur_lambda);
    auto cond_bias = std::abs(partial_deriv_bias) > cur_lambda;

    if (nc::sum(cond_weight)(0, 0) != 0 || cond_bias) {
      strong_set = nc::logical_or(strong_set, cond_weight);
    } else {
      KKT_holds = true;
    }
  }
}

void AdamTrainer::run(size_t batch_idx) {
  auto& design_matrix = this->_hal.design_matrix();
  auto outputs = this->_bdm.batchedMV(batch_idx, this->_hal.weights()); 
  (*outputs) += this->_hal.bias();

  nc::NdArray<float> batched_loss_weight = nc::empty<float>(0, 0); 
  if (this->_loss_weight.shape().rows != 0) {
    batched_loss_weight = this->_loss_weight(
      nc::Slice(
        this->_batched_start(batch_idx),
        this->_batched_end(batch_idx)
      ), 0
    );
  }

  auto grad = this->_loss.grad(*outputs,
                               this->_label(nc::Slice(this->_batched_start(batch_idx),
                                                      this->_batched_end(batch_idx)),
                                                      0),
                               batched_loss_weight).transpose(); 

  auto grad_weights = this->_bdm.batchedMV(batch_idx, grad, true);
  auto grad_bias = nc::sum(grad)(0, 0);

  this->_u_weights = this->_beta_1 * this->_u_weights + (1 - this->_beta_1) * (*grad_weights);
  this->_v_weights = this->_beta_2 * this->_v_weights + (1 - this->_beta_2) * nc::square(*grad_weights);
  this->_u_bias = this->_beta_1 * this->_u_bias + (1 - this->_beta_1) * grad_bias;
  this->_v_bias = this->_beta_2 * this->_v_bias + (1 - this->_beta_2) * nc::square(grad_bias);

  auto delta_weights = -(this->_step_size * this->_u_weights) / (nc::sqrt(this->_v_weights) + (float)1e-9);
  auto delta_bias = -(this->_step_size * this->_u_bias) / (nc::sqrt(this->_v_bias) + (float)1e-9);

  auto new_weights = _soft_threshold(this->_hal.weights() + delta_weights,
                                     this->_lambda, this->_step_size);
  auto new_bias = _soft_threshold(this->_hal.bias() + delta_bias,
                                  this->_lambda, this->_step_size);

  this->_hal.set_weights(new_weights);
  this->_hal.set_bias(new_bias);
}

size_t AdamTrainer::_batched_start(size_t batch_idx) const {
  auto batch_size = this->_batch_size;
  auto batch_start = batch_idx * batch_size;
  return batch_start;
}

size_t AdamTrainer::_batched_end(size_t batch_idx) const {
  auto batch_size = this->_batch_size;
  auto batch_end = std::min((batch_idx + 1) * batch_size, this->_bdm.nrow());
  return batch_end;
}


std::unique_ptr<nc::NdArray<float>> Predictor::predict(const nc::NdArray<float>& new_data) const {
  auto& design_matrix = this->_hal.design_matrix();
  auto pred_design_matrix = design_matrix.getPredDesignMatrix(new_data);
  auto outputs = pred_design_matrix->fusedRegionMV(
    0, pred_design_matrix->get_nrow(),
    0, pred_design_matrix->get_ncol(),
    this->_hal.weights()
  );
  (*outputs) += this->_hal.bias();
  return outputs;
}
