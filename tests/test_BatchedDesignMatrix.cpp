#include <NumCpp.hpp>
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

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
