#include <NumCpp.hpp>
#include <NumCpp/Functions/dot.hpp>
#include <NumCpp/Functions/norm.hpp>
#include <NumCpp/Random/randN.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include "DesignMatrix.hpp"
#include "BatchedDesignMatrix.hpp"
#include "BinSpMV.hpp"

TEST(BatchedDesignMatrix, RandomDf) {
  size_t max_order = 3;
  const size_t nrow = 50;
  const size_t ncol = 10; 
  auto df = nc::random::randN<float>({nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);
  auto batched_design_matrix = BatchedDesignMatrix(design_matrix, 6);

  auto expected = *design_matrix.getBatch(0, nrow)->full();

  std::vector<nc::NdArray<bool>> vec_res;
  for (int i = 0; i < batched_design_matrix.len(); i++) {
    vec_res.push_back(*batched_design_matrix.get(i)->full());
  }
  auto res = nc::stack(vec_res, nc::Axis::ROW);

  EXPECT_EQ(nc::sum(res - expected)(0, 0), 0);
}

TEST(BatchedDesignMatrix, prefetch) {
  size_t max_order = 3;
  const size_t nrow = 50;
  const size_t ncol = 10; 
  auto df = nc::random::randN<float>({nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);
  auto batched_design_matrix = BatchedDesignMatrix(design_matrix, 6);

  auto expected = *design_matrix.getBatch(0, nrow)->full();

  std::vector<nc::NdArray<bool>> vec_res;
  auto length = batched_design_matrix.len();
  for (int i = 0; i < length; i++) {
    vec_res.push_back(*batched_design_matrix.get(i)->full());
    if (i + 1 < length) {
      batched_design_matrix.prefetch(i + 1);
    }
  }
  auto res = nc::stack(vec_res, nc::Axis::ROW);

  EXPECT_EQ(nc::sum(res - expected)(0, 0), 0);
}

TEST(BatchedDesignMatrix, batchedMV) {
  size_t max_order = 3;
  const size_t nrow = 50;
  const size_t ncol = 10; 
  auto df = nc::random::randN<float>({nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);
  auto batched_design_matrix = BatchedDesignMatrix(design_matrix, 6);

  for (int i = 0; i < batched_design_matrix.len(); i++) {
    auto batched_data = batched_design_matrix.get(i)->full();
    auto rand_vec = nc::random::randN<float>({batched_data->shape().cols, 1});

    auto expected = nc::dot(batched_data->astype<float>(), rand_vec);
    auto res = batched_design_matrix.batchedMV(i, rand_vec);
    
    auto err = nc::norm(*res - expected)(0, 0) / nc::norm(expected)(0, 0);
    EXPECT_LE(err, 1e-5);
  }
}

TEST(BatchedDesignMatrix, batchedMVTranspose) {
  size_t max_order = 3;
  const size_t nrow = 50;
  const size_t ncol = 10; 
  auto df = nc::random::randN<float>({nrow, ncol});
  auto design_matrix = DesignMatrix(df, max_order);
  auto batched_design_matrix = BatchedDesignMatrix(design_matrix, 6);

  for (int i = 0; i < batched_design_matrix.len(); i++) {
    auto batched_data = batched_design_matrix.get(i)->full()->transpose();
    auto rand_vec = nc::random::randN<float>({batched_data.shape().cols, 1});

    auto expected = nc::dot(batched_data.astype<float>(), rand_vec);
    auto res = batched_design_matrix.batchedMV(i, rand_vec, true);
    
    auto err = nc::norm(*res - expected)(0, 0) / nc::norm(expected)(0, 0);
    EXPECT_LE(err, 1e-5);
  }
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
