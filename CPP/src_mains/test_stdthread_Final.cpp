#include "../include_for_cas766/PCH.hpp"
#include "../include_for_cas766/io_data.hpp"
#include "../include_for_cas766/matrix_convert.hpp"
#include "../include/parallel_stdthread_Final.hpp"
#include "../include/serial_spmv.hpp"

int main()
{
    MATRIX_CONVERT mat_convert;
    IO_DATA read_matrix(IO_MODE::UNSAFE);

    uint32_t num_thread = 4;
    double alpha = 1.0;
    double beta = 0.0;

    std::string file_path = "../../Test_Samples/mawi_201512020030.mtx";
    COO *coo_form = nullptr;
    coo_form = new COO();
    read_matrix.load_mat(file_path, *coo_form);

    D_VECTOR vec_x;
    vec_x.vec_element = std::vector<double>(coo_form->mat_data.num_col, 1);
    vec_x.vec_data.len_vec = coo_form->mat_data.num_col;

    D_VECTOR vec_b_coo;
    vec_b_coo.vec_element = std::vector<double>(coo_form->mat_data.num_col, 0);
    benchmark_spmv_serial(alpha, *coo_form, vec_x.vec_element, beta, vec_b_coo.vec_element, MAT_TYPE::coo);
    benchmark_spmv_parallel<COO>(alpha, *coo_form, vec_x.vec_element, beta, vec_b_coo.vec_element, num_thread, MAT_TYPE::coo, SYNC_TYPE::mutex_var);
    benchmark_spmv_parallel<COO>(alpha, *coo_form, vec_x.vec_element, beta, vec_b_coo.vec_element, num_thread, MAT_TYPE::coo, SYNC_TYPE::mutex_arr);
    benchmark_spmv_parallel<COO>(alpha, *coo_form, vec_x.vec_element, beta, vec_b_coo.vec_element, num_thread, MAT_TYPE::coo, SYNC_TYPE::atomic);

    CSC csc_form;
    mat_convert.convert(*coo_form, csc_form);
    D_VECTOR vec_b_csc;
    vec_b_csc.vec_element = std::vector<double>(csc_form.mat_data.num_col, 0);
    benchmark_spmv_serial(alpha, csc_form, vec_x.vec_element, beta, vec_b_csc.vec_element, MAT_TYPE::csc);
    benchmark_spmv_parallel<CSC>(alpha, csc_form, vec_x.vec_element, beta, vec_b_csc.vec_element, num_thread, MAT_TYPE::csc, SYNC_TYPE::mutex_var);
    benchmark_spmv_parallel<CSC>(alpha, csc_form, vec_x.vec_element, beta, vec_b_csc.vec_element, num_thread, MAT_TYPE::csc, SYNC_TYPE::mutex_arr);
    benchmark_spmv_parallel<CSC>(alpha, csc_form, vec_x.vec_element, beta, vec_b_csc.vec_element, num_thread, MAT_TYPE::csc, SYNC_TYPE::atomic);

    CSR csr_form;
    mat_convert.convert(*coo_form, csr_form);
    D_VECTOR vec_b_csr;
    vec_b_csr.vec_element = std::vector<double>(csr_form.mat_data.num_col, 0);
    benchmark_spmv_serial(alpha, csr_form, vec_x.vec_element, beta, vec_b_csr.vec_element, MAT_TYPE::csr);
    benchmark_spmv_parallel<CSR>(alpha, csr_form, vec_x.vec_element, beta, vec_b_csr.vec_element, num_thread, MAT_TYPE::csr, SYNC_TYPE::none);

    delete coo_form;
    coo_form = nullptr;

    return 0;
}
