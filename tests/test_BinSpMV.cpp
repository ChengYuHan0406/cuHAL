#include "BinSpMV.hpp"
#include "NumCpp.hpp"
#include <gtest/gtest.h>

TEST(BinSpMVTest, RandomBinMatrix) {
  const size_t nrow = 50;
  const size_t ncol = 20;
  auto randBinMat = nc::random::randInt<int>({nrow, ncol}, 0, 2).astype<bool>();
  auto binspmat = BinSpMat(nrow, ncol);

 /* Fills the `binspmat` at positions where `randBinMat` is true */
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      if (randBinMat(i, j) == true) {
        binspmat.fill(i, j);
      }
    }
  }

  /* Converts the data to Compressed Sparse Row (CSR) format */
  binspmat.translate();
  auto& row_ptr = binspmat.row_ptrs;
  auto& col_indices = binspmat.col_indices;

  /*
   * (1) Verifies that the number of non-zero elements in
   * `binspmat` is equal to that in `randBinMat` 
   */
  EXPECT_EQ(binspmat.nnz, nc::sum(randBinMat.astype<int>())(0, 0));

  /* (2) Ensures that if `binspmat[i, j]` is true, then `randBinMat[i, j]` is also true */
  for (int r = 0; r < nrow; r++) {
    for (int j = row_ptr[r]; j < row_ptr[r + 1]; j++) {
      auto c = col_indices[j]; 
      EXPECT_EQ(randBinMat(r, c), true);
    }
  }

 /* If both (1) and (2) hold, then `binspmat` and `randBinMat` must be identical */
}

TEST(BinSpMVTest, full) {
  const size_t nrow = 50;
  const size_t ncol = 20;
  auto randBinMat = nc::random::randInt<int>({nrow, ncol}, 0, 2).astype<bool>();
  auto binspmat = BinSpMat(nrow, ncol);

  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      if (randBinMat(i, j) == true) {
        binspmat.fill(i, j);
      }
    }
  }

  binspmat.translate();

  EXPECT_EQ(nc::sum(*binspmat.full() - randBinMat)(0, 0), 0);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
