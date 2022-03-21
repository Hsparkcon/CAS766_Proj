#include "../include_for_cas766/PCH.hpp"
#include "../include_for_cas766/io_data.hpp"
#include "../include_for_cas766/matrix_convert.hpp"
#include "../include/serial_spmv.hpp"
#include "../include/spmv_by_struct.hpp"
#include "../include/parallel_stdthread_Final.hpp"

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

    arr_COO *arr_coo = nullptr;
    arr_coo = new arr_COO();
    struct_convert(*coo_form, *arr_coo);

    D_VECTOR vec_x;
    vec_x.vec_element = std::vector<double>(coo_form->mat_data.num_col, 1);
    vec_x.vec_data.len_vec = coo_form->mat_data.num_col;

    D_VECTOR vec_b_coo;
    vec_b_coo.vec_element = std::vector<double>(coo_form->mat_data.num_col, 0);
    vec_b_coo.vec_data.len_vec = coo_form->mat_data.num_col;
    benchmark_spmv_serial<COO>(alpha, *coo_form, vec_x.vec_element, beta, vec_b_coo.vec_element, MAT_TYPE::coo);
    benchmark_serial_spmv_arr<arr_COO>(alpha, *arr_coo, vec_x.vec_element, beta, vec_b_coo.vec_element, MAT_TYPE::coo);
    benchmark_spmv_parallel<COO>(alpha, *coo_form, vec_x.vec_element, beta, vec_b_coo.vec_element, num_thread, MAT_TYPE::coo, SYNC_TYPE::atomic);
    benchmark_spmv_parallel_arr<arr_COO>(alpha, *arr_coo, vec_x.vec_element, beta, vec_b_coo.vec_element, num_thread, MAT_TYPE::coo);
    delete arr_coo;
    arr_coo = nullptr;

    CSC *csc_form = nullptr;
    csc_form = new CSC();
    mat_convert.convert(*coo_form, *csc_form);
    arr_CSC *arr_csc = nullptr;
    arr_csc = new arr_CSC();
    struct_convert(*csc_form, *arr_csc);
    D_VECTOR vec_b_csc;
    vec_b_csc.vec_element = std::vector<double>(coo_form->mat_data.num_col, 0);
    vec_b_csc.vec_data.len_vec = coo_form->mat_data.num_col;
    benchmark_spmv_serial<CSC>(alpha, *csc_form, vec_x.vec_element, beta, vec_b_csc.vec_element, MAT_TYPE::csc);
    benchmark_serial_spmv_arr<arr_CSC>(alpha, *arr_csc, vec_x.vec_element, beta, vec_b_csc.vec_element, MAT_TYPE::csc);
    benchmark_spmv_parallel<CSC>(alpha, *csc_form, vec_x.vec_element, beta, vec_b_coo.vec_element, num_thread, MAT_TYPE::csc, SYNC_TYPE::atomic);
    benchmark_spmv_parallel_arr<arr_CSC>(alpha, *arr_csc, vec_x.vec_element, beta, vec_b_coo.vec_element, num_thread, MAT_TYPE::csc);
    delete csc_form;
    csc_form = nullptr;
    delete arr_csc;
    arr_csc = nullptr;

    CSR *csr_form = nullptr;
    csr_form = new CSR();
    mat_convert.convert(*coo_form, *csr_form);
    arr_CSR *arr_csr = nullptr;
    arr_csr = new arr_CSR();
    struct_convert(*csr_form, *arr_csr);
    D_VECTOR vec_b_csr;
    vec_b_csr.vec_element = std::vector<double>(coo_form->mat_data.num_col, 0);
    vec_b_csr.vec_data.len_vec = coo_form->mat_data.num_col;
    benchmark_spmv_serial<CSR>(alpha, *csr_form, vec_x.vec_element, beta, vec_b_csr.vec_element, MAT_TYPE::csr);
    benchmark_serial_spmv_arr<arr_CSR>(alpha, *arr_csr, vec_x.vec_element, beta, vec_b_csr.vec_element, MAT_TYPE::csr);
    benchmark_spmv_parallel<CSR>(alpha, *csr_form, vec_x.vec_element, beta, vec_b_csr.vec_element, num_thread, MAT_TYPE::csr, SYNC_TYPE::none);
    benchmark_spmv_parallel_arr<arr_CSR>(alpha, *arr_csr, vec_x.vec_element, beta, vec_b_coo.vec_element, num_thread, MAT_TYPE::csr);
    delete csr_form;
    csr_form = nullptr;
    delete arr_csr;
    arr_csr = nullptr;

    delete coo_form;
    coo_form = nullptr;

    return 0;
}