#include "../include_for_cas766/PCH.hpp"
#include "../include_for_cas766/io_data.hpp"
#include "../include_for_cas766/matrix_convert.hpp"
#include "../include/demo.hpp"

int main()
{
    MATRIX_CONVERT mat_convert;
    IO_DATA read_matrix(IO_MODE::UNSAFE);

    uint32_t num_thread = 4;
    double alpha = 1.0;
    double beta = 0.0;

    std::string file_path = "../../Test_Samples/494_bus.mtx";
    COO *coo_form = nullptr;
    coo_form = new COO();
    read_matrix.load_mat(file_path, *coo_form);

    D_VECTOR vec_x;
    vec_x.vec_element = std::vector<double>(coo_form->mat_data.num_col, 1);
    vec_x.vec_data.len_vec = coo_form->mat_data.num_col;

    CSR csr_form;
    mat_convert.convert(*coo_form, csr_form);
    D_VECTOR vec_b_csr;
    vec_b_csr.vec_element = std::vector<double>(coo_form->mat_data.num_col, 0);
    vec_b_csr.vec_data.len_vec = coo_form->mat_data.num_col;

    openmp_spmv_CSR_static(alpha, csr_form, vec_x.vec_element, beta, vec_b_csr.vec_element, num_thread);

    delete coo_form;
    coo_form = nullptr;

    return 0;
}
