#include "HAL.hpp"
#include "DesignMatrix.hpp"
#include <NumCpp.hpp>
#include "assert.h"
#include <cmath>
#include "omp.h"
#include <cstdlib>
#include <stdlib.h> 
#include <time.h>

std::unique_ptr<nc::NdArray<float>> batch_binspmv(BatchedDesignMatrix& A, const nc::NdArray<float>& x);

HAL::HAL(const nc::NdArray<float>& dataframe,
         const nc::NdArray<float>& labels,
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
  auto grad = this->_loss.grad(*outputs, this->_label, this->_loss_weight).transpose(); 

  return grad;
}

std::unique_ptr<nc::NdArray<float>> SRTrainer::partial_derivs(const nc::NdArray<float>& out_grad) {
  auto& design_matrix = this->_hal.design_matrix();

  auto partial_derivs = design_matrix.fusedRegionMV(
    0, this->_hal.design_matrix().get_nrow(),
    0, this->_hal.design_matrix().get_ncol(),
    out_grad,
    true
  );

  return partial_derivs;
}

float SRTrainer::partial_deriv_bias(const nc::NdArray<float>& out_grad) {
  return nc::sum(out_grad)(0, 0);
}

SRTrainer::SRTrainer(HAL& hal,
                     const float step_size,
                     const Loss& loss,
                     const nc::NdArray<float> loss_weight,
                     const size_t max_iters,
                     const float beta_1,
                     const float beta_2) : _hal(hal), 
                                           _loss(loss),
                                           _step_size(step_size),
                                           _loss_weight(loss_weight),
                                           _max_iters(max_iters),
                                           _label(hal.labels()),
                                           _epsilon(1e-6),
                                           _num_lambdas(100),
                                           _beta_1(beta_1),
                                           _beta_2(beta_2),
                                           _u_weights(nc::zeros<float>({hal.weights().shape().rows, 1})),
                                           _v_weights(nc::zeros<float>({hal.weights().shape().rows, 1})),
                                           _u_bias(0),
                                           _v_bias(0) {
  
  auto& design_matrix = this->_hal.design_matrix();
  auto out_grad = this->grad_wrt_outputs();
  auto partial_grad = this->partial_derivs(out_grad); 
  this->_lambda_max = nc::max(nc::abs(*partial_grad))(0, 0);
  this->_lambda_min = this->_epsilon * this->_lambda_max;
  this->_lambda_step = std::pow(this->_epsilon,
                                -(1 / (float)(this->_num_lambdas - 1)));
} 

void SRTrainer::run(const nc::NdArray<float>& val_df,
                    const nc::NdArray<float>& val_label,
                    const nc::NdArray<float>& val_loss_weight) {
  auto predictor = Predictor(this->_hal);

  float cur_lambda = this->_lambda_max;
  float prev_lambda = 0;

  float best_loss = std::numeric_limits<float>::max();
  float best_lambda;
  nc::NdArray<float> best_weights; 
  float best_bias;

  for (int i = 0; i < this->_num_lambdas; i++) {
    std::cout << "Lambda = " << cur_lambda << std::endl;
    this->solve_lambda(cur_lambda, prev_lambda);

    auto val_out = *predictor.predict(val_df);
    auto val_loss = this->_loss.compute(val_out, val_label, val_loss_weight);
    std::cout << "Validation Loss: " << val_loss << std::endl;

    if (val_loss < best_loss) {
      best_loss = val_loss;
      best_weights = this->_hal.weights();
      best_bias = this->_hal.bias();
      best_lambda = cur_lambda;
    }

    prev_lambda = cur_lambda;
    cur_lambda = prev_lambda / this->_lambda_step;
  }

  std::cout << "========================== Results =============================" << std::endl;
  std::cout << "Best Lambda: " << best_lambda << std::endl;
  std::cout << "Best Validation Loss: " << best_loss << std::endl << std::endl;
  std::cout << "Saving HAL Weights..." << std::endl;

  this->_hal.set_weights(best_weights);
  this->_hal.set_bias(best_bias);
}

void SRTrainer::solve_lambda(float cur_lambda, float prev_lambda) {
  auto& design_matrix = this->_hal.design_matrix();

  auto thres = 2 * cur_lambda - prev_lambda;
  auto nonzeros = (this->_hal.weights() != 0.0f);
  auto out_grad = this->grad_wrt_outputs();
  auto strong_set = (nc::abs(*this->partial_derivs(out_grad)) > thres);
  strong_set = nc::logical_or(strong_set, nonzeros);
  bool include_bias = false;

  bool KKT_holds = false;
  
  int iters = 0;
  while (!KKT_holds && iters++ < this->_max_iters) {
    auto out_grad = this->grad_wrt_outputs();

    if (include_bias) {
      auto cur_grad = this->partial_deriv_bias(out_grad);

      this->_u_bias = this->_beta_1 * this->_u_bias + (1 - this->_beta_1) * cur_grad;
      this->_v_bias = this->_beta_2 * this->_v_bias + (1 - this->_beta_2) * nc::square(cur_grad);

      auto delta_bias = -(this->_step_size * this->_u_bias) / (nc::sqrt(this->_v_bias) + (float)1e-9);
      auto new_bias = _soft_threshold(this->_hal.bias() + delta_bias,
                                      cur_lambda, this->_step_size);

      this->_hal.set_bias(new_bias);
    }

    auto len_strong_set = nc::sum(strong_set.astype<int>())(0, 0); 

    if (len_strong_set) {
      auto colidx_subset
          = nc::arange<int>(0, design_matrix.get_ncol())[strong_set].reshape(1, len_strong_set);
      auto partial_grad = design_matrix.fusedColSubsetMV(out_grad, colidx_subset, true);

      for (int i = 0; i < len_strong_set; i++) {
        auto cur_colidx = colidx_subset(0, i);
        auto cur_grad = (*partial_grad)(cur_colidx, 0);
        auto cur_u = this->_u_weights(cur_colidx, 0);
        auto cur_v = this->_v_weights(cur_colidx, 0);

        this->_u_weights(cur_colidx, 0)
          = this->_beta_1 * cur_u + (1 - this->_beta_1) * cur_grad;
        this->_v_weights(cur_colidx, 0)
          = this->_beta_2 * cur_v + (1 - this->_beta_2) * nc::square(cur_grad);
      }

      auto delta_weights = -(this->_step_size * this->_u_weights) / (nc::sqrt(this->_v_weights) + (float)1e-9);
      auto new_weights = _soft_threshold(this->_hal.weights() + delta_weights,
                                        cur_lambda, this->_step_size);

      this->_hal.set_weights(new_weights);
    }

    /* Check KKT */
    out_grad = this->grad_wrt_outputs();
    auto partial_deriv_weight = this->partial_derivs(out_grad);
    auto partial_deriv_bias = this->partial_deriv_bias(out_grad);

    auto cond_weight = (nc::abs(*partial_deriv_weight) > cur_lambda);
    auto cond_bias = std::abs(partial_deriv_bias) > cur_lambda;

    if (nc::sum(cond_weight.astype<int>())(0, 0) != 0 || cond_bias) {
      strong_set = nc::logical_or(strong_set, cond_weight);
      include_bias = cond_bias;
    } else {
      KKT_holds = true;
      std::cout << "KKT holds" << std::endl;
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
