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
         float sample_ratio) : _design_matrix(DesignMatrix(dataframe, max_order, sample_ratio)),
                               _labels(labels),
                               _max_order(max_order),
                               _sample_ratio(sample_ratio) {

  auto shape_dataframe = dataframe.shape();
  auto shape_labels = labels.shape();

  /* Check data shape */
  assert(shape_labels.cols == 1);
  assert(shape_dataframe.rows == shape_labels.rows);

  auto nrow = this->_design_matrix.get_nrow();
  auto ncol = this->_design_matrix.get_ncol();
  
  /* Initialize `_weights` an `_bias` */
  srand(time(NULL));
  float sigma = 1 / sqrt(nrow * ncol);
  this->_weights = sigma * nc::random::randN<float>({static_cast<uint32_t>(ncol), 1}); 
  this->_bias = sigma * nc::random::randN<float>({1, 1})(0, 0);
}

void HAL::update_weights(const size_t idx, const float delta) {
  if (idx == 0) {
    this->_bias += delta;
  } else {
    this->_weights(idx - 1, 0) += delta;
  }
}

void PSCDTrainer::run_one_iteration() {
  auto outputs = batch_binspmv(this->_batched_design_matrix, this->_hal.weights());
  (*outputs) += this->_hal.bias();
  auto grad = this->_loss.grad(*outputs, this->_label); 

  size_t num_threads = omp_get_max_threads();
  std::vector<std::pair<size_t, float>> deltas(num_threads); // vector of (column idx, delta)

  srand(time(NULL));
  #pragma omp parallel num_threads(num_threads)
  {
    size_t thread_idx = omp_get_thread_num();
    /* Number of columns including bias term */
    auto ncol = this->_batched_design_matrix.ncol() + 1;
    auto col_idx = rand() % ncol;

    float partial_grad;
    float weight;

    if (col_idx != 0) {
      auto cur_col = this->_hal.design_matrix().getCol(col_idx - 1);
      weight = this->_hal.weights()(col_idx - 1, 0);
      partial_grad = nc::dot(grad, cur_col->astype<float>())(0, 0);
      partial_grad += this->_lambda * nc::sign(weight);
    } else {
      weight = this->_hal.bias();
      partial_grad = nc::sum(grad)(0, 0);
      partial_grad += this->_lambda * nc::sign(weight);
    }

    float delta = std::max(-partial_grad * this->_step_size, -weight);
    deltas[thread_idx] = std::pair<size_t, float>(col_idx, delta);
  }

  for (auto p : deltas) {
    this->_hal.update_weights(p.first, p.second);
  }
}

std::unique_ptr<nc::NdArray<float>> Predictor::predict(const nc::NdArray<float>& new_data) const {
  auto& design_matrix = this->_hal.design_matrix();
  auto pred_design_matrix = design_matrix.getPredDesignMatrix(new_data);
  auto batched_pred_design_matrix = BatchedDesignMatrix(*pred_design_matrix, this->_batch_size);
  auto outputs = batch_binspmv(batched_pred_design_matrix, this->_hal.weights());
  (*outputs) += this->_hal.bias();
  return outputs;
}
