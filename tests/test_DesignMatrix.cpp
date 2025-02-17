#include "BatchedDesignMatrix.hpp"
#include "DesignMatrix.hpp"
#include <NumCpp.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>

nc::NdArray<bool> static _create_block() {
  nc::NdArray<bool> res = {
      {true, false, false}, {true, true, false}, {true, true, true}};
  return res;
}

std::unique_ptr<nc::NdArray<float>> batch_binspmv(BatchedDesignMatrix &A,
                                                  const nc::NdArray<float> &x);

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
      auto &col_index = design_matrix.ColIndices[col_count];
      EXPECT_EQ(col_index.interaction.size(), 1);
      EXPECT_EQ(col_index.interaction[0], i);
      EXPECT_EQ(col_index.sample_idx, s);
      col_count++;
    }
  }
  // Check order 2
  for (int i = 0; i < ncol; i++) {
    for (int j = i + 1; j < ncol; j++) {
      for (int s = 0; s < nrow; s += 2) {
        auto &col_index = design_matrix.ColIndices[col_count];
        EXPECT_EQ(col_index.interaction.size(), 2);
        EXPECT_EQ(col_index.interaction[0], i);
        EXPECT_EQ(col_index.interaction[1], j);
        EXPECT_EQ(col_index.sample_idx, s);
        col_count++;
      }
    }
  }
}

bool _find_colindex(const std::vector<size_t> &interact, size_t sample_idx,
                    const DesignMatrix &dm) {

  auto &ColIndices = dm.ColIndices;
  for (auto col : ColIndices) {
    auto &cur_interact = col.interaction;
    auto cur_sample_idx = col.sample_idx;

    bool same_interact = (interact.size() == cur_interact.size());
    for (int i = 0; i < std::min(interact.size(), cur_interact.size()); i++) {
      same_interact &= (interact[i] == cur_interact[i]);
    }

    bool same = (sample_idx == cur_sample_idx) & same_interact;
    if (same) {
      return true;
    }
  }
  return false;
}

TEST(DesignMatrixTest, InitColIndicesRemoveDup) {
  size_t max_order = 2;
  const size_t nrow = 100;
  const size_t ncol = 20;

  auto df = nc::random::randInt<int>({nrow, ncol}, 0, 10).astype<float>();
  auto design_matrix = DesignMatrix(df, max_order);

  // Check order 1
  for (int i = 0; i < ncol; i++) {
    CutPoints cut_points;
    for (int s = 0; s < nrow; s++) {
      std::vector<float> cur_cut_point;
      cur_cut_point.push_back(df(s, i));
      bool duplicate = (cut_points.find(cur_cut_point) != cut_points.end());
      if (!_find_colindex({(size_t)i}, s, design_matrix)) {
        EXPECT_EQ(duplicate, true);
      } else {
        EXPECT_EQ(duplicate, false);
      }
      cut_points.insert(cur_cut_point);
    }
  }
  // Check order 2
  for (int i = 0; i < ncol; i++) {
    for (int j = i + 1; j < ncol; j++) {
      CutPoints cut_points;
      for (int s = 0; s < nrow; s += 2) {
        std::vector<float> cur_cut_point;
        cur_cut_point.push_back(df(s, i));
        cur_cut_point.push_back(df(s, j));
        bool duplicate = (cut_points.find(cur_cut_point) != cut_points.end());
        if (!_find_colindex({(size_t)i, (size_t)j}, s, design_matrix)) {
          EXPECT_EQ(duplicate, true);
        } else {
          EXPECT_EQ(duplicate, false);
        } 
        cut_points.insert(cur_cut_point);
      }
    }
  }
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

      auto &col_index = design_matrix.ColIndices[j];
      auto &interact = col_index.interaction;
      auto sample_idx = col_index.sample_idx;

      bool expected = true;
      for (auto v : interact) {
        expected &= (df(global_row_idx, v) >= df(sample_idx, v));
      }

      EXPECT_EQ(cur_val, expected);
    }
  }
}

TEST(DesignMatrixTest, getPredDesignMatrix) {
  size_t max_order = 3;
  const size_t nrow = 100;
  const size_t test_nrow = 20;
  const size_t ncol = 50;
  auto df = nc::random::randN<float>({nrow, ncol});
  auto test_df = nc::random::randN<float>({test_nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);
  auto pred_design_matrix = design_matrix.getPredDesignMatrix(test_df);
  auto realized_pred_design_matrix =
      pred_design_matrix->getBatch(0, test_nrow)->full();

  auto& col_indices = design_matrix.ColIndices;
  for (int r = 0; r < test_nrow; r++) {
    for (int c = 0; c < design_matrix.get_ncol(); c++) {
      auto& cur_col = col_indices[c];
      auto& interact = cur_col.interaction;
      auto sample_idx = cur_col.sample_idx;

      bool cur_val = true;
      for (auto x : interact) {
        cur_val &= (test_df(r, x) >= df(sample_idx, x));
      }
      EXPECT_EQ(cur_val, (*realized_pred_design_matrix)(r, c));
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
  auto pred_design_matrix_1 =
      design_matrix.getPredDesignMatrix(test_df_1.astype<float>());
  auto res_1 = pred_design_matrix_1->getBatch(0, nrow)->full();
  auto expected_1 = nc::ones<bool>({3, (uint32_t)design_matrix.get_ncol()});
  EXPECT_EQ(nc::sum(*res_1 - expected_1)(0, 0), 0);

  auto test_df_2 = nc::arange<int>(-9, 0).reshape(nrow, ncol);
  auto pred_design_matrix_2 =
      design_matrix.getPredDesignMatrix(test_df_2.astype<float>());
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
  auto realized_pred_design_matrix =
      pred_design_matrix->getBatch(0, test_nrow)->full();

  for (int i = 0; i < pred_design_matrix->get_nrow(); i++) {
    for (int j = 0; j < pred_design_matrix->get_ncol(); j++) {
      auto &col_index = pred_design_matrix->ColIndices[j];
      auto &interact = col_index.interaction;
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

TEST(DesignMatrixTest, getRegion) {
  size_t max_order = 3;
  const size_t nrow = 50;
  const size_t ncol = 10;
  auto df = nc::random::randN<float>({nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);

  auto row_start = 10, row_end = 20;
  auto col_start = 5, col_end = 8;
  std::unique_ptr<nc::NdArray<bool>> realized_region =
      design_matrix.getRegion(row_start, row_end, col_start, col_end)->full();

  auto row_size = row_end - row_start;
  auto col_size = col_end - col_start;

  for (int i = 0; i < row_size; i++) {
    for (int j = 0; j < col_size; j++) {
      bool cur_val = (*realized_region)(i, j);

      auto global_row_idx = row_start + i;
      auto global_col_idx = col_start + j;

      auto &col_index = design_matrix.ColIndices[global_col_idx];
      auto &interact = col_index.interaction;
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

  auto rand_vec =
      nc::random::randN<float>({(uint32_t)(col_end - col_start), 1});

  auto expected =
      nc::dot(design_matrix.getRegion(row_start, row_end, col_start, col_end)
                  ->full()
                  ->astype<float>(),
              rand_vec);

  auto res = design_matrix.fusedRegionMV(row_start, row_end, col_start, col_end,
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

  auto rand_vec =
      nc::random::randN<float>({(uint32_t)(row_end - row_start), 1});

  auto expected =
      nc::dot(design_matrix.getRegion(row_start, row_end, col_start, col_end)
                  ->full()
                  ->astype<float>()
                  .transpose(),
              rand_vec);

  auto res = design_matrix.fusedRegionMV(row_start, row_end, col_start, col_end,
                                         rand_vec, true);

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

  auto rand_vec =
      nc::random::randN<float>({(uint32_t)design_matrix.get_ncol(), 1});
  auto realized_pred_design_matrix =
      pred_design_matrix->getBatch(0, test_nrow)->full();

  auto expected = nc::dot(realized_pred_design_matrix->astype<float>(), rand_vec);
  auto res = pred_design_matrix->fusedRegionMV(
      0, pred_design_matrix->get_nrow(), 0, pred_design_matrix->get_ncol(),
      rand_vec);

  auto err = nc::norm(*res - expected)(0, 0) / nc::norm(expected)(0, 0);
  EXPECT_LE(err, 1e-5);
}

TEST(DesignMatrixTest, proportion_ones) {
  size_t max_order = 3;
  const size_t nrow = 50;
  const size_t ncol = 10;
  auto df = nc::random::randN<float>({nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);

  auto full_dm = design_matrix.getBatch(0, nrow)->full();
  auto expected = nc::mean(full_dm->astype<float>(), nc::Axis::ROW)
                      .transpose()
                      .astype<float>();

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
        (*proportion_ones)(c, 0) > 0.0f) {
      vec_reduce_mat.push_back((*full_dm)(full_dm->rSlice(), c));
    }
  }

  auto expected_reduce_mat = nc::stack(vec_reduce_mat, nc::Axis::COL);

  design_matrix.reduce_basis(epsilon);
  auto reduce_mat = design_matrix.getBatch(0, design_matrix.get_nrow())->full();

  auto err = nc::norm(*reduce_mat - expected_reduce_mat)(0, 0) /
             nc::norm(expected_reduce_mat)(0, 0);
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
        (*proportion_ones)(c, 0) > 0.0f) {
      vec_reduce_mat.push_back((*full_dm)(full_dm->rSlice(), c));
    }
  }

  auto expected_reduce_mat = nc::stack(vec_reduce_mat, nc::Axis::COL);
  auto rand_vec =
      nc::random::randN<float>({expected_reduce_mat.shape().cols, 1});

  auto expected_res = nc::dot(expected_reduce_mat.astype<float>(), rand_vec);

  design_matrix.reduce_basis(epsilon);
  auto res = design_matrix.fusedRegionMV(0, nrow, 0, design_matrix.get_ncol(),
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
        (*proportion_ones)(c, 0) > 0.0f) {
      vec_reduce_mat.push_back((*full_dm)(full_dm->rSlice(), c));
    }
  }

  auto expected_reduce_mat =
      nc::stack(vec_reduce_mat, nc::Axis::COL).transpose();
  auto rand_vec =
      nc::random::randN<float>({expected_reduce_mat.shape().cols, 1});

  auto expected_res = nc::dot(expected_reduce_mat.astype<float>(), rand_vec);

  design_matrix.reduce_basis(epsilon);
  auto res = design_matrix.fusedRegionMV(0, nrow, 0, design_matrix.get_ncol(),
                                         rand_vec, true);

  auto err = nc::norm(*res - expected_res)(0, 0) / nc::norm(expected_res)(0, 0);
  EXPECT_LE(err, 1e-5);
}

TEST(DesignMatrixTest, fusedColSubsetMV) {
  size_t max_order = 2;
  const size_t nrow = 100;
  const size_t ncol = 50;
  const size_t size_colsubset = 10;
  auto df = nc::random::randN<float>({nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);
  auto full_dm = design_matrix.getBatch(0, nrow)->full();
  auto rand_vec = nc::random::randN<float>({size_colsubset, 1});

  auto dm_ncol = design_matrix.get_ncol();
  auto rand_colidx = nc::random::randInt<int>({1, size_colsubset}, 0, dm_ncol);

  std::vector<nc::NdArray<bool>> vec_cols;
  for (int c = 0; c < size_colsubset; c++) {
    vec_cols.push_back((*full_dm)(full_dm->rSlice(), rand_colidx(0, c)));
  }

  auto expected_dm = nc::stack(vec_cols, nc::Axis::COL);
  auto expected_res = nc::dot(expected_dm.astype<float>(), rand_vec);
  auto res = design_matrix.fusedColSubsetMV(rand_vec, rand_colidx);

  auto err = nc::norm(*res - expected_res)(0, 0) / nc::norm(expected_res)(0, 0);
  EXPECT_LE(err, 1e-5);
}

TEST(DesignMatrixTest, fusedColSubsetMVTranspose) {
  size_t max_order = 2;
  const size_t nrow = 100;
  const size_t ncol = 50;
  const size_t size_colsubset = 10;
  auto df = nc::random::randN<float>({nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);
  auto full_dm = design_matrix.getBatch(0, nrow)->full();
  auto rand_vec = nc::random::randN<float>({nrow, 1});

  auto dm_ncol = design_matrix.get_ncol();
  auto rand_colidx = nc::random::randInt<int>({1, size_colsubset}, 0, dm_ncol);

  std::vector<nc::NdArray<bool>> vec_cols;
  for (int c = 0; c < size_colsubset; c++) {
    vec_cols.push_back((*full_dm)(full_dm->rSlice(), rand_colidx(0, c)));
  }

  auto expected_dm = nc::stack(vec_cols, nc::Axis::COL);
  auto expected_res =
      nc::dot(expected_dm.astype<float>().transpose(), rand_vec);
  auto res = design_matrix.fusedColSubsetMV(rand_vec, rand_colidx, true);

  auto err = nc::norm(*res - expected_res)(0, 0) / nc::norm(expected_res)(0, 0);
  EXPECT_LE(err, 1e-5);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
