#include "HAL.hpp"
#include "DesignMatrix.hpp"
#include <NumCpp.hpp>
#include "assert.h"
#include <NumCpp/Functions/clip.hpp>
#include <NumCpp/Functions/maximum.hpp>
#include <NumCpp/Functions/zeros.hpp>
#include <cmath>
#include "omp.h"
#include <stdlib.h> 
#include <time.h>

std::unique_ptr<nc::NdArray<float>> batch_binspmv(BatchedDesignMatrix& A, const nc::NdArray<float>& x);

HAL::HAL(const nc::NdArray<float>& dataframe,
         nc::NdArray<float> labels,
         size_t max_order,
         float sample_ratio) : _design_matrix(dataframe, max_order, sample_ratio),
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

void GDTrainer::run(size_t batch_start, size_t batch_end) {
  auto& design_matrix = this->_hal.design_matrix();
  auto outputs = design_matrix.fusedRegionMV(
    0, this->_hal.design_matrix().get_nrow(),
    0, this->_hal.design_matrix().get_ncol(),
    this->_hal.weights()
  );
  (*outputs) += this->_hal.bias();
  auto grad = this->_loss.grad(*outputs, this->_label).transpose(); 

  auto grad_weights = design_matrix.fusedRegionMV(
                        batch_start,
                        batch_end,
                        0,
                        design_matrix.get_ncol(),
                        grad,
                        true
                      );

  auto grad_bias = nc::sum(grad)(0, 0);

  auto new_weights = _soft_threshold(this->_hal.weights() - this->_step_size * (*grad_weights),
                                     this->_lambda, this->_step_size);
  auto new_bias = _soft_threshold(this->_hal.bias() - this->_step_size * grad_bias,
                                  this->_lambda, this->_step_size);

  this->_hal.set_weights(new_weights);
  this->_hal.set_bias(new_bias);
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
