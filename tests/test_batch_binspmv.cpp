#include "DesignMatrix.hpp"
#include <NumCpp.hpp>
#include <gtest/gtest.h>
#include <memory>
#include "BatchedDesignMatrix.hpp"
#include "DesignMatrix.hpp"
#include "BinSpMV.hpp"

std::unique_ptr<nc::NdArray<float>> batch_binspmv(BatchedDesignMatrix& A, const nc::NdArray<float>& x); 

TEST(batchbinspmvTest, RandomMV) {
  const size_t df_ncol = 10;
  const size_t df_nrow = 50;
  auto df = nc::random::randN<float>({df_nrow, df_ncol});
  auto A = DesignMatrix(df, 2);
  auto batched_A = BatchedDesignMatrix(A, 2);
  const uint32_t ncol = A.get_ncol();
  auto x = nc::random::randN<float>({ncol, 1});

  auto realized_A = A.getBatch(0, A.get_nrow())->full();
  auto expected = nc::dot(realized_A->astype<float>(), x);
  auto res = batch_binspmv(batched_A, x);

  EXPECT_LE((nc::norm(*res - expected)(0, 0) / nc::norm(expected)(0, 0)), 1e-5);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
