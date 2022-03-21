#include "../include_for_cas766/PCH.hpp"
#include "../include_for_cas766/matrix_struct.hpp"

void dense_to_csr(const D_MATRIX &d_mat, CSR &s_mat)
{
    s_mat.mat_data = d_mat.mat_data;
    s_mat.row_ptr = std::vector<uint32_t>(d_mat.mat_data.num_row + 1);
    s_mat.col_and_val = std::vector<IDX_VAL>(d_mat.mat_data.num_nz);

    uint32_t track_nz = 0;
    for (uint32_t iter_row = 0; iter_row < d_mat.mat_data.num_row; ++iter_row)
    {
        uint32_t nnz_in_row = 0;
        for (uint32_t iter_col = 0; iter_col < d_mat.mat_data.num_col; ++iter_col)
        {
            if (d_mat.matrix[iter_row][iter_col] != 0)
            {
                s_mat.col_and_val[track_nz].idx = iter_col;
                s_mat.col_and_val[track_nz].val = d_mat.matrix[iter_row][iter_col];
                nnz_in_row++;
                track_nz++;
            }
        }
        s_mat.row_ptr[iter_row + 1] = nnz_in_row;
    }

    s_mat.row_ptr[0] = 0;
    for (uint32_t iter_r_ptr = 1; iter_r_ptr < s_mat.row_ptr.size(); ++iter_r_ptr)
    {
        s_mat.row_ptr[iter_r_ptr] = s_mat.row_ptr[iter_r_ptr] + s_mat.row_ptr[iter_r_ptr - 1];
    }
}

void dense_to_csc(const D_MATRIX &d_mat, CSC &s_mat)
{
    s_mat.mat_data = d_mat.mat_data;
    s_mat.col_ptr = std::vector<uint32_t>(d_mat.mat_data.num_col + 1);
    s_mat.row_and_val = std::vector<IDX_VAL>(d_mat.mat_data.num_nz);

    uint32_t track_nz = 0;
    for (uint32_t iter_col = 0; iter_col < d_mat.mat_data.num_col; ++iter_col)
    {
        uint32_t nnz_in_col = 0;
        for (uint32_t iter_row = 0; iter_row < d_mat.mat_data.num_row; ++iter_row)
        {

            if (d_mat.matrix[iter_row][iter_col] != 0)
            {
                s_mat.row_and_val[track_nz].idx = iter_row;
                s_mat.row_and_val[track_nz].val = d_mat.matrix[iter_row][iter_col];
                nnz_in_col++;
                track_nz++;
            }
        }
        s_mat.col_ptr[iter_col + 1] = nnz_in_col;
    }

    s_mat.col_ptr[0] = 0;
    for (uint32_t iter_c_ptr = 1; iter_c_ptr < s_mat.col_ptr.size(); ++iter_c_ptr)
    {
        s_mat.col_ptr[iter_c_ptr] = s_mat.col_ptr[iter_c_ptr] + s_mat.col_ptr[iter_c_ptr - 1];
    }
}

void csr_to_dense(const CSR &s_mat, D_MATRIX &d_mat)
{
    d_mat.mat_data = s_mat.mat_data;

    d_mat.matrix =
        std::vector<std::vector<double>>(
            s_mat.mat_data.num_row, std::vector<double>(s_mat.mat_data.num_col, 0.0));

    for (uint32_t idx_r_ptr = 0; idx_r_ptr < s_mat.mat_data.num_row; ++idx_r_ptr)
    {
        uint32_t nz_start = s_mat.row_ptr[idx_r_ptr];
        uint32_t nz_end = s_mat.row_ptr[idx_r_ptr + 1];

        for (uint32_t idx_nz = nz_start; idx_nz < nz_end; ++idx_nz)
        {
            d_mat.matrix[idx_r_ptr][s_mat.col_and_val[idx_nz].idx] =
                s_mat.col_and_val[idx_nz].val;
        }
    }
}