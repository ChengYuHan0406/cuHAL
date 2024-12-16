#include "NumCpp.hpp"
#include <gtest/gtest.h>
#include <memory>
#include "DesignMatrix.hpp"

nc::NdArray<bool> static _create_block() {
  nc::NdArray<bool> res =  {{true, false, false},
                           {true, true, false},
                           {true, true, true}};
  return res;
}

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
      auto col_index = design_matrix.ColIndices[col_count];
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
        auto col_index = design_matrix.ColIndices[col_count];
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
      auto col = design_matrix.getCol(col_indices[i]);
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
  auto err = nc::sum(*batch1 - expected_DesignMatrix(nc::Slice(start_idx1, end_idx1),
                                                     nc::Slice(0, 18)))(0, 0);
  EXPECT_EQ(err, 0);

  size_t start_idx2 = 2, end_idx2 = 3;
  auto batch2 = design_matrix.getBatch(start_idx2, end_idx2);
  auto err2 = nc::sum(*batch2 - expected_DesignMatrix(nc::Slice(start_idx2, end_idx2),
                                                      nc::Slice(0, 18)))(0, 0);
  EXPECT_EQ(err2, 0);

  size_t start_idx_full = 0, end_idx_full = 3;
  auto batch_full = design_matrix.getBatch(start_idx_full, end_idx_full);
  auto err_full = nc::sum(*batch_full - expected_DesignMatrix(nc::Slice(start_idx_full, end_idx_full),
                                                              nc::Slice(0, 18)))(0, 0);
  EXPECT_EQ(err_full, 0);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


