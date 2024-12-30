#include <NumCpp.hpp>
#include <NumCpp/Core/Enums.hpp>
#include <NumCpp/Core/Slice.hpp>
#include <NumCpp/Functions/dot.hpp>
#include <NumCpp/Functions/norm.hpp>
#include <NumCpp/Functions/ones.hpp>
#include <NumCpp/Functions/stack.hpp>
#include <NumCpp/Functions/sum.hpp>
#include <NumCpp/Random/randN.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include "DesignMatrix.hpp"
#include "BatchedDesignMatrix.hpp"

nc::NdArray<bool> static _create_block() {
  nc::NdArray<bool> res =  {{true, false, false},
                           {true, true, false},
                           {true, true, true}};
  return res;
}

std::unique_ptr<nc::NdArray<float>> batch_binspmv(BatchedDesignMatrix& A, const nc::NdArray<float>& x);

TEST(DesignMatrixTest, InitColIndices) {
  size_t max_order = 2;
  size_t nrow = 20;
  size_t ncol = 10;
  auto df = nc::arange<int>(0, nrow * ncol).reshape(nrow, ncol);
  auto design_matrix = DesignMatrix(df.astype<float>(), max_order);

  size_t col_count = 0;
  // Check order 1
  for (int i = 0; i < ncol; i++) {
    for (int s = 0; s < nrow; s++) {
      auto& col_index = design_matrix.ColIndices[col_count];
      EXPECT_EQ(col_index.interaction.size(), 1);
      EXPECT_EQ(col_index.interaction[0], i);
      EXPECT_EQ(col_index.sample_idx, s);
      col_count++;
    }
  }
  // Check order 2
  for (int i = 0; i < ncol; i++) {
    for (int j = i + 1; j < ncol; j++) {
      for (int s = 0; s < nrow; s++) {
        auto& col_index = design_matrix.ColIndices[col_count];
        EXPECT_EQ(col_index.interaction.size(), 2);
        EXPECT_EQ(col_index.interaction[0], i);
        EXPECT_EQ(col_index.interaction[1], j);
        EXPECT_EQ(col_index.sample_idx, s);
        col_count++;
      }
    }
  }
}

/*********************************** Test DataFrame *********************************************** 
        ===============
        df
        ===============
         V1 | V2 | V3 
        ===============
     0    0  | 1  | 2
        ---------------
     1    3  | 4  | 5
        ---------------
     2    6  | 7  | 8
        ===============

        ===============================================================
        Order 1 Support Points
        ===============================================================
         V1_0 | V1_1 | V1_2 | V2_0 | V2_1 | V2_2 | V3_0 | V3_1 | V3_2
        ===============================================================
          0   |  3   |   6  |   1  |   4  |  7   |   2  |   5  |   8       
        ===============================================================

        ========================================================================================
        Order 2 Support Points
        ========================================================================================
         V1_V2_0 | V1_V2_1 | V1_V2_2 | V1_V3_0 | V1_V3_1 | V1_V3_2 | V2_V3_0 | V2_V3_l | V2_V3_2
        ========================================================================================
          (0, 1) | (3, 4)  | (6, 7)  | (0, 2)  | (3, 5)  | (6, 8)  | (1, 2)  | (4, 5)  | (7, 8)
        ========================================================================================

        Design Matrix
        ===============================================================    
         V1_0 | V1_1 | V1_2 | V2_0 | V2_1 | V2_2 | V3_0 | V3_1 | V3_2
        ===============================================================
    0      1  |  0   |  0   |   1  |  0   |  0   |   1  |  0   |  0   |
        ---------------------------------------------------------------
    1      1  |  1   |  0   |   1  |  1   |  0   |   1  |  1   |  0   |
        ---------------------------------------------------------------
    2      1  |  1   |  1   |   1  |  1   |  1   |   1  |  1   |  1   |
        =========================================================================================
         V1_V2_0 | V1_V2_1 | V1_V2_2 | V1_V3_0 | V1_V3_1 | V1_V3_2 | V2_V3_0 | V2_V3_l | V2_V3_2
        =========================================================================================
    0      1     |    0    |    0    |     1   |    0    |    0    |     1   |    0    |    0   |
        -----------------------------------------------------------------------------------------
    1      1     |    1    |    0    |     1   |    1    |    0    |     1   |    1    |    0   |
        -----------------------------------------------------------------------------------------
    2      1     |    1    |    1    |     1   |    1    |    1    |     1   |    1    |    1   |
        =========================================================================================

**************************************************************************************************************************/


TEST(DesignMatrixTest, getCol) {
  size_t max_order = 2;
  size_t nrow = 3;
  size_t ncol = 3;
  auto df = nc::arange<int>(0, 9).reshape(nrow, ncol);
  auto design_matrix = DesignMatrix(df.astype<float>(), max_order);
  auto expected_DesignMatrix = nc::stack({_create_block(), _create_block(),_create_block(),
                             _create_block(), _create_block(),_create_block()},
                             nc::Axis::COL);

  auto col_indices = design_matrix.ColIndices;
  for (int i = 0; i < col_indices.size(); i++) {
      auto col = design_matrix.getCol(i);
      auto expected = expected_DesignMatrix(expected_DesignMatrix.rSlice(), i);

      EXPECT_EQ(col->shape().cols, expected.shape().cols);
      EXPECT_EQ(col->shape().rows, expected.shape().rows);
      EXPECT_EQ(nc::sum(*col - expected)(0, 0), 0);
  }
}

TEST(DesignMatrixTest, getBatch) {

  size_t max_order = 2;
  size_t nrow = 3;
  size_t ncol = 3;
  auto df = nc::arange<int>(0, 9).reshape(nrow, ncol);
  auto design_matrix = DesignMatrix(df.astype<float>(), max_order);
  auto expected_DesignMatrix = nc::stack({_create_block(), _create_block(),_create_block(),
                             _create_block(), _create_block(),_create_block()},
                             nc::Axis::COL);

  size_t start_idx1 = 0, end_idx1 = 2;
  auto batch1 = design_matrix.getBatch(start_idx1, end_idx1);
  auto err = nc::sum(*batch1->full() - expected_DesignMatrix(nc::Slice(start_idx1, end_idx1),
                                                     nc::Slice(0, 18)))(0, 0);

  EXPECT_EQ(err, 0);

  size_t start_idx2 = 2, end_idx2 = 3;
  auto batch2 = design_matrix.getBatch(start_idx2, end_idx2);
  auto err2 = nc::sum(*batch2->full() - expected_DesignMatrix(nc::Slice(start_idx2, end_idx2),
                                                      nc::Slice(0, 18)))(0, 0);

  EXPECT_EQ(err2, 0);

  size_t start_idx_full = 0, end_idx_full = 3;
  auto batch_full = design_matrix.getBatch(start_idx_full, end_idx_full);
  auto err_full = nc::sum(*batch_full->full() - expected_DesignMatrix(nc::Slice(start_idx_full, end_idx_full),
                                                              nc::Slice(0, 18)))(0, 0);

  EXPECT_EQ(err_full, 0);
}

TEST(DesignMatrixTest, RandomDfgetBatch) {
  size_t max_order = 3;
  const size_t nrow = 100;
  const size_t ncol = 20;
  auto df = nc::random::randN<float>({nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);

  auto row_start = 10, row_end = 50;
  auto realized_batch = design_matrix.getBatch(row_start, row_end)->full();

  auto row_size = row_end - row_start;

  for (int i = 0; i < row_size; i++) {
    for (int j = 0; j < design_matrix.get_ncol(); j++) {
      bool cur_val = (*realized_batch)(i, j);
      
      auto global_row_idx = row_start + i;

      auto& col_index = design_matrix.ColIndices[j];
      auto& interact = col_index.interaction;
      auto sample_idx = col_index.sample_idx;

      bool expected = true;
      for (auto v : interact) {
        expected &= (df(global_row_idx, v) >= df(sample_idx, v));
      }

      EXPECT_EQ(cur_val, expected);
    }
  }
}

TEST(DesignMatrixTest, getBatchPred) {
  size_t max_order = 2;
  size_t nrow = 3;
  size_t ncol = 3;
  auto df = nc::arange<int>(0, 9).reshape(nrow, ncol);
  auto design_matrix = DesignMatrix(df.astype<float>(), max_order);

  auto test_df_1 = nc::arange<int>(9, 18).reshape(nrow, ncol);
  auto pred_design_matrix_1 = design_matrix.getPredDesignMatrix(test_df_1.astype<float>());
  auto res_1 = pred_design_matrix_1->getBatch(0, nrow)->full();
  auto expected_1 = nc::ones<bool>({3, (uint32_t)design_matrix.get_ncol()});
  EXPECT_EQ(nc::sum(*res_1 - expected_1)(0, 0), 0);

  auto test_df_2 = nc::arange<int>(-9, 0).reshape(nrow, ncol);
  auto pred_design_matrix_2 = design_matrix.getPredDesignMatrix(test_df_2.astype<float>());
  auto res_2 = pred_design_matrix_2->getBatch(0, nrow)->full();
  auto expected_2 = nc::zeros<bool>({3, (uint32_t)design_matrix.get_ncol()});
  EXPECT_EQ(nc::sum(*res_2 - expected_2)(0, 0), 0);
}

TEST(DesignMatrixTest, RandomDfgetBatchPred) {
  size_t max_order = 3;
  const size_t nrow = 100;
  const size_t test_nrow = 10;
  const size_t ncol = 20;
  auto df = nc::random::randN<float>({nrow, ncol});
  auto test_df = nc::random::randN<float>({test_nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);
  auto pred_design_matrix = design_matrix.getPredDesignMatrix(test_df);
  auto realized_pred_design_matrix = pred_design_matrix->getBatch(0, test_nrow)->full();

  for (int i = 0; i < pred_design_matrix->get_nrow(); i++) {
    for (int j = 0; j < pred_design_matrix->get_ncol(); j++) {
      auto& col_index = pred_design_matrix->ColIndices[j]; 
      auto& interact = col_index.interaction;
      auto sample_idx = col_index.sample_idx;
      
      bool expected = true;
      for (auto c : interact) {
        auto thres = df(sample_idx, c);
        auto val = test_df(i, c);
        expected &= (val >= thres);
      }

      EXPECT_EQ((*realized_pred_design_matrix)(i, j), expected);
    }
  }
}

TEST(DesignMatrixTest, BatchedDesignMatrix) {
  size_t max_order = 2;
  size_t nrow = 3;
  size_t ncol = 3;
  auto df = nc::arange<int>(0, 9).reshape(nrow, ncol);
  auto design_matrix = DesignMatrix(df.astype<float>(), max_order);
  auto expected_DesignMatrix = nc::stack({_create_block(), _create_block(),_create_block(),
                             _create_block(), _create_block(),_create_block()},
                             nc::Axis::COL);

  // batch_size = 1
  auto batched_design_matrix_1 = BatchedDesignMatrix(design_matrix, 1);
  EXPECT_EQ(batched_design_matrix_1.len(), 3);
  int err1_0 = nc::sum(*batched_design_matrix_1.get(0)->full() - expected_DesignMatrix(0, nc::Slice(0, 18)))(0, 0);
  int err1_1 = nc::sum(*batched_design_matrix_1.get(1)->full() - expected_DesignMatrix(1, nc::Slice(0, 18)))(0, 0);
  int err1_2 = nc::sum(*batched_design_matrix_1.get(2)->full() - expected_DesignMatrix(2, nc::Slice(0, 18)))(0, 0);
  EXPECT_EQ(err1_0, 0);
  EXPECT_EQ(err1_1, 0);
  EXPECT_EQ(err1_2, 0);

  // batch_size = 2
  auto batched_design_matrix_2 = BatchedDesignMatrix(design_matrix, 2);
  EXPECT_EQ(batched_design_matrix_2.len(), 2);
  int err2_0 = nc::sum(*batched_design_matrix_2.get(0)->full() - expected_DesignMatrix(nc::Slice(0, 2), nc::Slice(0, 18)))(0, 0);
  int err2_1 = nc::sum(*batched_design_matrix_2.get(1)->full() - expected_DesignMatrix(2, nc::Slice(0, 18)))(0, 0);
  EXPECT_EQ(err2_0, 0);
  EXPECT_EQ(err2_1, 0);

  // batch_size = 3
  auto batched_design_matrix_3 = BatchedDesignMatrix(design_matrix, 3);
  EXPECT_EQ(batched_design_matrix_3.len(), 1);
  int err3 = nc::sum(*batched_design_matrix_3.get(0)->full() - expected_DesignMatrix)(0, 0);
}

TEST(DesignMatrixTest, getRegion) {
  size_t max_order = 3;
  const size_t nrow = 50;
  const size_t ncol = 10; 
  auto df = nc::random::randN<float>({nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);

  auto row_start = 10, row_end = 20;
  auto col_start = 5, col_end = 8;
  std::unique_ptr<nc::NdArray<bool>> realized_region = design_matrix.getRegion(row_start,
                                                                               row_end,
                                                                               col_start,
                                                                               col_end)->full();

  auto row_size = row_end - row_start;
  auto col_size = col_end - col_start;

  for (int i = 0; i < row_size; i++) {
    for (int j = 0; j < col_size; j++) {
      bool cur_val = (*realized_region)(i, j);
      
      auto global_row_idx = row_start + i;
      auto global_col_idx = col_start + j;

      auto& col_index = design_matrix.ColIndices[global_col_idx];
      auto& interact = col_index.interaction;
      auto sample_idx = col_index.sample_idx;

      bool expected = true;
      for (auto v : interact) {
        expected &= (df(global_row_idx, v) >= df(sample_idx, v));
      }

      EXPECT_EQ(cur_val, expected);
    }
  }
}

TEST(DesignMatrixTest, fusedRegionMV) {
  size_t max_order = 3;
  const size_t nrow = 50;
  const size_t ncol = 10; 
  auto df = nc::random::randN<float>({nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);

  auto row_start = 10, row_end = 20;
  auto col_start = 5, col_end = 8;

  auto rand_vec = nc::random::randN<float>({(uint32_t)(col_end - col_start), 1});

  auto expected = nc::dot(design_matrix.getRegion(row_start,
                                                  row_end,
                                                  col_start,
                                                  col_end)->full()->astype<float>(),
                          rand_vec);

  auto res = design_matrix.fusedRegionMV(row_start,
                                         row_end,
                                         col_start,
                                         col_end,
                                         rand_vec);

  auto err = nc::norm(*res - expected)(0, 0) / nc::norm(expected)(0, 0);
  EXPECT_LE(err, 1e-5);
}

TEST(DesignMatrixTest, fusedRegionMVTranspose) {
  size_t max_order = 3;
  const size_t nrow = 50;
  const size_t ncol = 10; 
  auto df = nc::random::randN<float>({nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);

  auto row_start = 10, row_end = 20;
  auto col_start = 5, col_end = 8;

  auto rand_vec = nc::random::randN<float>({(uint32_t)(row_end - row_start), 1});

  auto expected = nc::dot(design_matrix.getRegion(row_start,
                                                  row_end,
                                                  col_start,
                                                  col_end)->full()->astype<float>().transpose(),
                          rand_vec);

  auto res = design_matrix.fusedRegionMV(row_start,
                                         row_end,
                                         col_start,
                                         col_end,
                                         rand_vec,
                                         true);

  auto err = nc::norm(*res - expected)(0, 0) / nc::norm(expected)(0, 0);
  EXPECT_LE(err, 1e-5);
}

TEST(DesignMatrixTest, fusedRegionMVPred) {
  size_t max_order = 3;
  const size_t nrow = 100;
  const size_t test_nrow = 10;
  const size_t ncol = 20;
  auto df = nc::random::randN<float>({nrow, ncol});
  auto test_df = nc::random::randN<float>({test_nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);
  auto pred_design_matrix = design_matrix.getPredDesignMatrix(test_df);

  auto rand_vec = nc::random::randN<float>({(uint32_t)design_matrix.get_ncol(), 1});
  auto bdm = BatchedDesignMatrix(*pred_design_matrix, 100);

  auto expected = batch_binspmv(bdm, rand_vec);
  auto res = pred_design_matrix->fusedRegionMV(0,
                                               pred_design_matrix->get_nrow(),
                                               0,
                                               pred_design_matrix->get_ncol(), 
                                               rand_vec);

  auto err = nc::norm(*res - *expected)(0, 0) / nc::norm(*expected)(0, 0);
  EXPECT_LE(err, 1e-5);
}

TEST(DesignMatrixTest, proportion_ones) {
  size_t max_order = 3;
  const size_t nrow = 50;
  const size_t ncol = 10; 
  auto df = nc::random::randN<float>({nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);

  auto full_dm = design_matrix.getBatch(0, nrow)->full();
  auto expected = nc::mean(full_dm->astype<float>(), nc::Axis::ROW).transpose().astype<float>();

  auto res = design_matrix.proportion_ones();
  auto err = nc::norm(*res - expected)(0, 0) / nc::norm(expected)(0, 0);

  EXPECT_LE(err, 1e-5);
}

TEST(DesignMatrixTest, reduce_basis) {
  size_t max_order = 2;
  const size_t nrow = 100;
  const size_t ncol = 50; 
  auto df = nc::random::randN<float>({nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);
  auto full_dm = design_matrix.getBatch(0, nrow)->full();

  auto proportion_ones = design_matrix.proportion_ones();
  float epsilon = 0.001;
  std::vector<nc::NdArray<bool>> vec_reduce_mat;

  float min_except_zero = 2;
  for (int c = 0; c < proportion_ones->shape().rows; c++) {
    auto cur_val = (*proportion_ones)(c, 0);
    if (cur_val < min_except_zero && cur_val != 0) {
      min_except_zero = cur_val; 
    }
  }

  float lower_bound = min_except_zero * (1 + epsilon);

  for (int c = 0; c < full_dm->shape().cols; c++) {
    if ((*proportion_ones)(c, 0) < 1.0f &&
        (*proportion_ones)(c, 0) >= lower_bound &&
        (*proportion_ones)(c, 0) > 0.0f
       ) {
      vec_reduce_mat.push_back((*full_dm)(full_dm->rSlice(), c));
    }
  }

  auto expected_reduce_mat = nc::stack(vec_reduce_mat, nc::Axis::COL);

  design_matrix.reduce_basis(epsilon);
  auto reduce_mat = design_matrix.getBatch(0, design_matrix.get_nrow())->full();

  auto err = nc::norm(*reduce_mat - expected_reduce_mat)(0, 0) / nc::norm(expected_reduce_mat)(0, 0);
  EXPECT_LE(err, 1e-5);
}

TEST(DesignMatrixTest, fusedRegionMVAfterReduceBasis) {
  size_t max_order = 2;
  const size_t nrow = 100;
  const size_t ncol = 50; 
  auto df = nc::random::randN<float>({nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);
  auto full_dm = design_matrix.getBatch(0, nrow)->full();

  auto proportion_ones = design_matrix.proportion_ones();
  float epsilon = 0.001;
  std::vector<nc::NdArray<bool>> vec_reduce_mat;

  float min_except_zero = 2;
  for (int c = 0; c < proportion_ones->shape().rows; c++) {
    auto cur_val = (*proportion_ones)(c, 0);
    if (cur_val < min_except_zero && cur_val != 0) {
      min_except_zero = cur_val; 
    }
  }

  float lower_bound = min_except_zero * (1 + epsilon);

  for (int c = 0; c < full_dm->shape().cols; c++) {
    if ((*proportion_ones)(c, 0) < 1.0f &&
        (*proportion_ones)(c, 0) >= lower_bound &&
        (*proportion_ones)(c, 0) > 0.0f
       ) {
      vec_reduce_mat.push_back((*full_dm)(full_dm->rSlice(), c));
    }
  }
  
  auto expected_reduce_mat = nc::stack(vec_reduce_mat, nc::Axis::COL);
  auto rand_vec = nc::random::randN<float>({expected_reduce_mat.shape().cols, 1});

  auto expected_res = nc::dot(expected_reduce_mat.astype<float>(), rand_vec);
  
  design_matrix.reduce_basis(epsilon);
  auto res = design_matrix.fusedRegionMV(0,
                                         nrow,
                                         0,
                                         design_matrix.get_ncol(),
                                         rand_vec);

  auto err = nc::norm(*res - expected_res)(0, 0) / nc::norm(expected_res)(0, 0);
  EXPECT_LE(err, 1e-5);
}

TEST(DesignMatrixTest, fusedRegionMVTransposeAfterReduceBasis) {
  size_t max_order = 2;
  const size_t nrow = 100;
  const size_t ncol = 50; 
  auto df = nc::random::randN<float>({nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);
  auto full_dm = design_matrix.getBatch(0, nrow)->full();

  auto proportion_ones = design_matrix.proportion_ones();
  float epsilon = 0.001;
  std::vector<nc::NdArray<bool>> vec_reduce_mat;

  float min_except_zero = 2;
  for (int c = 0; c < proportion_ones->shape().rows; c++) {
    auto cur_val = (*proportion_ones)(c, 0);
    if (cur_val < min_except_zero && cur_val != 0) {
      min_except_zero = cur_val; 
    }
  }

  float lower_bound = min_except_zero * (1 + epsilon);

  for (int c = 0; c < full_dm->shape().cols; c++) {
    if ((*proportion_ones)(c, 0) < 1.0f &&
        (*proportion_ones)(c, 0) >= lower_bound &&
        (*proportion_ones)(c, 0) > 0.0f
       ) {
      vec_reduce_mat.push_back((*full_dm)(full_dm->rSlice(), c));
    }
  }
  
  auto expected_reduce_mat = nc::stack(vec_reduce_mat, nc::Axis::COL).transpose();
  auto rand_vec = nc::random::randN<float>({expected_reduce_mat.shape().cols, 1});

  auto expected_res = nc::dot(expected_reduce_mat.astype<float>(), rand_vec);
  
  design_matrix.reduce_basis(epsilon);
  auto res = design_matrix.fusedRegionMV(0,
                                         nrow,
                                         0,
                                         design_matrix.get_ncol(),
                                         rand_vec,
                                         true);

  auto err = nc::norm(*res - expected_res)(0, 0) / nc::norm(expected_res)(0, 0);
  EXPECT_LE(err, 1e-5);
}


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


