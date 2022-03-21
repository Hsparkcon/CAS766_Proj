package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

type MAT_INFO struct {
	num_row uint32
	num_col uint32
	num_nz  uint32
}

type TRIPLET struct {
	idx_row uint32
	idx_col uint32
	value   float64
}

type COO struct {
	mat_data MAT_INFO
	mat_elem []TRIPLET
}

type PAIR struct {
	idx   uint32
	value float64
}

type CSC struct {
	mat_data MAT_INFO
	col_ptr  []uint32
	mat_elem []PAIR
}

type CSR struct {
	mat_data MAT_INFO
	row_ptr  []uint32
	mat_elem []PAIR
}

func mtx_load(file_path string) COO {
	content, err := os.Open(file_path)
	if err != nil {
		log.Fatal(err)
	}

	var mat_A COO
	var mtx_read = bufio.NewReader(content)
	var read_line []byte
	for {
		read_line, _, _ = mtx_read.ReadLine()
		if string(read_line[0]) == "%" {
			continue
		}
		fmt.Fscanf(strings.NewReader(string(read_line)), "%d %d %d", &mat_A.mat_data.num_row, &mat_A.mat_data.num_col, &mat_A.mat_data.num_nz)
		break
	}

	mat_A.mat_elem = make([]TRIPLET, mat_A.mat_data.num_nz)

	var num_line uint32 = 0
	for ; num_line < mat_A.mat_data.num_nz; num_line++ {
		read_line, _, _ = mtx_read.ReadLine()
		fmt.Fscanf(strings.NewReader(string(read_line)), "%d %d %g", &mat_A.mat_elem[num_line].idx_row, &mat_A.mat_elem[num_line].idx_col, &mat_A.mat_elem[num_line].value)
		mat_A.mat_elem[num_line].idx_row -= 1
		mat_A.mat_elem[num_line].idx_col -= 1
	}
	content.Close()

	return mat_A
}

func COO_TO_CSC(mat_COO COO) CSC {
	sort.SliceStable(mat_COO.mat_elem, func(i, j int) bool {
		if mat_COO.mat_elem[i].idx_col > mat_COO.mat_elem[j].idx_col {
			return false
		} else if mat_COO.mat_elem[i].idx_col == mat_COO.mat_elem[j].idx_col &&
			mat_COO.mat_elem[i].idx_row > mat_COO.mat_elem[j].idx_row {
			return false
		} else {
			return true
		}
	})

	var mat CSC
	mat.mat_data = mat_COO.mat_data
	mat.col_ptr = make([]uint32, mat_COO.mat_data.num_col+1)
	mat.mat_elem = make([]PAIR, mat_COO.mat_data.num_nz)

	mat.col_ptr[0] = 0
	var ptr_tracker uint32 = 0
	var idx_nz uint32 = 0
	for ; idx_nz < mat_COO.mat_data.num_nz; idx_nz++ {
		for ; ptr_tracker < mat_COO.mat_elem[idx_nz].idx_col; ptr_tracker++ {
			mat.col_ptr[ptr_tracker+1] = idx_nz
		}
		mat.mat_elem[idx_nz].idx = mat_COO.mat_elem[idx_nz].idx_row
		mat.mat_elem[idx_nz].value = mat_COO.mat_elem[idx_nz].value
	}

	for ; ptr_tracker < mat_COO.mat_data.num_col-1; ptr_tracker++ {
		mat.col_ptr[ptr_tracker] = mat.mat_data.num_nz
	}

	mat.col_ptr[mat_COO.mat_data.num_col] = mat.mat_data.num_nz

	return mat
}

func COO_TO_CSR(mat_COO COO) CSR {
	sort.SliceStable(mat_COO.mat_elem, func(i, j int) bool {
		if mat_COO.mat_elem[i].idx_row > mat_COO.mat_elem[j].idx_row {
			return false
		} else if mat_COO.mat_elem[i].idx_row == mat_COO.mat_elem[j].idx_row &&
			mat_COO.mat_elem[i].idx_col > mat_COO.mat_elem[j].idx_col {
			return false
		} else {
			return true
		}
	})

	var mat CSR
	mat.mat_data = mat_COO.mat_data
	mat.row_ptr = make([]uint32, mat_COO.mat_data.num_row+1)
	mat.mat_elem = make([]PAIR, mat_COO.mat_data.num_nz)

	mat.row_ptr[0] = 0
	var ptr_tracker uint32 = 0
	var idx_nz uint32 = 0
	for ; idx_nz < mat_COO.mat_data.num_nz; idx_nz++ {
		for ; ptr_tracker < mat_COO.mat_elem[idx_nz].idx_row; ptr_tracker++ {
			mat.row_ptr[ptr_tracker+1] = idx_nz
		}
		mat.mat_elem[idx_nz].idx = mat_COO.mat_elem[idx_nz].idx_col
		mat.mat_elem[idx_nz].value = mat_COO.mat_elem[idx_nz].value
	}

	for ; ptr_tracker < mat_COO.mat_data.num_row-1; ptr_tracker++ {
		mat.row_ptr[ptr_tracker] = mat.mat_data.num_nz
	}

	mat.row_ptr[mat_COO.mat_data.num_row] = mat.mat_data.num_nz

	return mat
}

func gen_sample_vec(mat_A COO, init_val float64) []float64 {
	var vec_len uint32 = mat_A.mat_data.num_col
	var vec = make([]float64, vec_len)

	for i := 0; i < int(vec_len); i++ {
		vec[i] = init_val
	}
	return vec
}

func coo_spmv_serial(alpha float64, mat_A COO, vec_x []float64, beta float64, vec_b []float64, num_thread uint32) {
	var idx_vec_b uint32 = 0
	for ; idx_vec_b < uint32(len(vec_x)); idx_vec_b++ {
		vec_b[idx_vec_b] = beta * vec_b[idx_vec_b]
	}

	var idx_nz uint32 = 0
	for ; idx_nz < mat_A.mat_data.num_nz; idx_nz++ {
		vec_b[mat_A.mat_elem[idx_nz].idx_row] += (alpha * mat_A.mat_elem[idx_nz].value * vec_x[mat_A.mat_elem[idx_nz].idx_col])
	}
}

func coo_spmv_parallel_mutex_var(alpha float64, mat_A COO, vec_x []float64, beta float64, vec_b []float64, num_thread uint32) {
	var work_range uint32 = mat_A.mat_data.num_nz / num_thread
	var idx_start_list = make([]uint32, num_thread)
	var idx_end_list = make([]uint32, num_thread)

	var idx_thread uint32 = 0
	for ; idx_thread < num_thread; idx_thread++ {
		idx_start_list[idx_thread] = idx_thread * work_range
		if idx_thread == num_thread-1 {
			idx_end_list[idx_thread] = mat_A.mat_data.num_nz
		} else {
			idx_end_list[idx_thread] = idx_start_list[idx_thread] + work_range
		}
	}

	var idx_vec_b uint32 = 0
	for ; idx_vec_b < uint32(len(vec_x)); idx_vec_b++ {
		vec_b[idx_vec_b] = beta * vec_b[idx_vec_b]
	}

	var wg sync.WaitGroup
	var mutex_var sync.Mutex
	wg.Add(int(num_thread))
	idx_thread = 0
	for ; idx_thread < num_thread; idx_thread++ {
		go func(idx_start uint32, idx_end uint32, mutex *sync.Mutex, wg *sync.WaitGroup) {
			var idx_nz uint32 = idx_start
			for ; idx_nz < idx_end; idx_nz++ {
				mutex.Lock()
				vec_b[mat_A.mat_elem[idx_nz].idx_row] += (alpha * mat_A.mat_elem[idx_nz].value * vec_x[mat_A.mat_elem[idx_nz].idx_col])
				mutex.Unlock()
			}
			wg.Done()
		}(idx_start_list[idx_thread], idx_end_list[idx_thread], &mutex_var, &wg)
	}
	wg.Wait()
}

func coo_spmv_parallel_mutex_arr(alpha float64, mat_A COO, vec_x []float64, beta float64, vec_b []float64, num_thread uint32) {
	var work_range uint32 = mat_A.mat_data.num_nz / num_thread
	var idx_start_list = make([]uint32, num_thread)
	var idx_end_list = make([]uint32, num_thread)

	var idx_thread uint32 = 0
	for ; idx_thread < num_thread; idx_thread++ {
		idx_start_list[idx_thread] = idx_thread * work_range
		if idx_thread == num_thread-1 {
			idx_end_list[idx_thread] = mat_A.mat_data.num_nz
		} else {
			idx_end_list[idx_thread] = idx_start_list[idx_thread] + work_range
		}
	}

	var idx_vec_b uint32 = 0
	for ; idx_vec_b < uint32(len(vec_x)); idx_vec_b++ {
		vec_b[idx_vec_b] = beta * vec_b[idx_vec_b]
	}

	var wg sync.WaitGroup
	var mut_arr []sync.Mutex
	mut_arr = make([]sync.Mutex, len(vec_b))
	wg.Add(int(num_thread))
	idx_thread = 0
	for ; idx_thread < num_thread; idx_thread++ {
		go func(idx_start uint32, idx_end uint32, mutex_arr []sync.Mutex, wg *sync.WaitGroup) {
			var idx_nz uint32 = idx_start
			for ; idx_nz < idx_end; idx_nz++ {
				mutex_arr[mat_A.mat_elem[idx_nz].idx_row].Lock()
				vec_b[mat_A.mat_elem[idx_nz].idx_row] += (alpha * mat_A.mat_elem[idx_nz].value * vec_x[mat_A.mat_elem[idx_nz].idx_col])
				mutex_arr[mat_A.mat_elem[idx_nz].idx_row].Unlock()
			}
			wg.Done()
		}(idx_start_list[idx_thread], idx_end_list[idx_thread], mut_arr[:], &wg)
	}
	wg.Wait()
}

func csc_spmv_serial(alpha float64, mat_A CSC, vec_x []float64, beta float64, vec_b []float64, num_thread uint32) {
	for idx_vec_b := 0; idx_vec_b < len(vec_b); idx_vec_b++ {
		vec_b[idx_vec_b] = beta * vec_b[idx_vec_b]
	}

	for col_ptr := 0; col_ptr < int(mat_A.mat_data.num_col); col_ptr++ {
		for idx_nz := mat_A.col_ptr[col_ptr]; idx_nz < mat_A.col_ptr[col_ptr+1]; idx_nz++ {
			vec_b[mat_A.mat_elem[idx_nz].idx] += (alpha * mat_A.mat_elem[idx_nz].value * vec_x[col_ptr])
		}
	}
}

func csc_spmv_parallel_mutex_var(alpha float64, mat_A CSC, vec_x []float64, beta float64, vec_b []float64, num_thread uint32) {
	for idx_vec_b := 0; idx_vec_b < len(vec_b); idx_vec_b++ {
		vec_b[idx_vec_b] = beta * vec_b[idx_vec_b]
	}

	var work_range uint32 = mat_A.mat_data.num_col / num_thread
	var idx_start_list = make([]uint32, num_thread)
	var idx_end_list = make([]uint32, num_thread)

	var idx_thread uint32 = 0
	for ; idx_thread < num_thread; idx_thread++ {
		idx_start_list[idx_thread] = idx_thread * work_range
		if idx_thread == num_thread-1 {
			idx_end_list[idx_thread] = mat_A.mat_data.num_col
		} else {
			idx_end_list[idx_thread] = idx_start_list[idx_thread] + work_range
		}
	}

	var wg sync.WaitGroup
	var mutex_var sync.Mutex
	wg.Add(int(num_thread))
	for idx_thread := 0; idx_thread < int(num_thread); idx_thread++ {
		go func(c_ptr_start uint32, c_ptr_end uint32, mutex *sync.Mutex, wg *sync.WaitGroup) {
			for col_ptr := c_ptr_start; col_ptr < c_ptr_end; col_ptr++ {
				for idx_nz := mat_A.col_ptr[col_ptr]; idx_nz < mat_A.col_ptr[col_ptr+1]; idx_nz++ {
					mutex.Lock()
					vec_b[mat_A.mat_elem[idx_nz].idx] += (alpha * mat_A.mat_elem[idx_nz].value * vec_x[col_ptr])
					mutex.Unlock()
				}
			}
			wg.Done()
		}(idx_start_list[idx_thread], idx_end_list[idx_thread], &mutex_var, &wg)
	}
	wg.Wait()
}

func csc_spmv_parallel_mutex_arr(alpha float64, mat_A CSC, vec_x []float64, beta float64, vec_b []float64, num_thread uint32) {
	for idx_vec_b := 0; idx_vec_b < len(vec_b); idx_vec_b++ {
		vec_b[idx_vec_b] = beta * vec_b[idx_vec_b]
	}

	var work_range uint32 = mat_A.mat_data.num_col / num_thread
	var idx_start_list = make([]uint32, num_thread)
	var idx_end_list = make([]uint32, num_thread)

	var idx_thread uint32 = 0
	for ; idx_thread < num_thread; idx_thread++ {
		idx_start_list[idx_thread] = idx_thread * work_range
		if idx_thread == num_thread-1 {
			idx_end_list[idx_thread] = mat_A.mat_data.num_col
		} else {
			idx_end_list[idx_thread] = idx_start_list[idx_thread] + work_range
		}
	}

	var wg sync.WaitGroup
	var mut_arr []sync.Mutex
	mut_arr = make([]sync.Mutex, len(vec_b))
	wg.Add(int(num_thread))
	for idx_thread := 0; idx_thread < int(num_thread); idx_thread++ {
		go func(c_ptr_start uint32, c_ptr_end uint32, mutex_arr []sync.Mutex, wg *sync.WaitGroup) {
			for col_ptr := c_ptr_start; col_ptr < c_ptr_end; col_ptr++ {
				for idx_nz := mat_A.col_ptr[col_ptr]; idx_nz < mat_A.col_ptr[col_ptr+1]; idx_nz++ {
					mutex_arr[mat_A.mat_elem[idx_nz].idx].Lock()
					vec_b[mat_A.mat_elem[idx_nz].idx] += (alpha * mat_A.mat_elem[idx_nz].value * vec_x[col_ptr])
					mutex_arr[mat_A.mat_elem[idx_nz].idx].Unlock()
				}
			}
			wg.Done()
		}(idx_start_list[idx_thread], idx_end_list[idx_thread], mut_arr[:], &wg)
	}
	wg.Wait()
}

func csr_spmv_serial(alpha float64, mat_A CSR, vec_x []float64, beta float64, vec_b []float64, num_thread uint32) {
	for row_ptr := 0; row_ptr < int(mat_A.mat_data.num_row); row_ptr++ {
		vec_b[row_ptr] = beta * vec_b[row_ptr]
		for idx_nz := mat_A.row_ptr[row_ptr]; idx_nz < mat_A.row_ptr[row_ptr+1]; idx_nz++ {
			vec_b[row_ptr] += (alpha * mat_A.mat_elem[idx_nz].value * vec_x[mat_A.mat_elem[idx_nz].idx])
		}
	}
}

func csr_spmv_parallel(alpha float64, mat_A CSR, vec_x []float64, beta float64, vec_b []float64, num_thread uint32) {
	var work_range uint32 = mat_A.mat_data.num_row / num_thread
	var idx_start_list = make([]uint32, num_thread)
	var idx_end_list = make([]uint32, num_thread)

	var idx_thread uint32 = 0
	for ; idx_thread < num_thread; idx_thread++ {
		idx_start_list[idx_thread] = idx_thread * work_range
		if idx_thread == num_thread-1 {
			idx_end_list[idx_thread] = mat_A.mat_data.num_row
		} else {
			idx_end_list[idx_thread] = idx_start_list[idx_thread] + work_range
		}
	}

	var wg sync.WaitGroup
	wg.Add(int(num_thread))
	idx_thread = 0
	for ; idx_thread < num_thread; idx_thread++ {
		go func(r_ptr_start uint32, r_ptr_end uint32, wg *sync.WaitGroup) {
			for row_ptr := r_ptr_start; row_ptr < r_ptr_end; row_ptr++ {
				vec_b[row_ptr] = beta * vec_b[row_ptr]
				for idx_nz := mat_A.row_ptr[row_ptr]; idx_nz < mat_A.row_ptr[row_ptr+1]; idx_nz++ {
					vec_b[row_ptr] += (alpha * mat_A.mat_elem[idx_nz].value * vec_x[mat_A.mat_elem[idx_nz].idx])
				}
			}
			wg.Done()
		}(idx_start_list[idx_thread], idx_end_list[idx_thread], &wg)
	}
	wg.Wait()
}

type fn_COO func(alpha float64, mat_A COO, vec_x []float64, beta float64, vec_b []float64, num_thread uint32)

func bench_COO(coo_func fn_COO, alpha float64, mat_A COO, vec_x []float64, beta float64, vec_b []float64, num_thread uint32, operation string) {
	var iter_num int = 10
	start := time.Now()
	for i := 0; i < iter_num; i++ {
		coo_func(alpha, mat_A, vec_x[:], beta, vec_b[:], num_thread)
	}
	elapsed := time.Since(start)
	avg_time := elapsed / time.Duration(iter_num)
	fmt.Println(operation, avg_time.Milliseconds())
}

type fn_CSC func(alpha float64, mat_A CSC, vec_x []float64, beta float64, vec_b []float64, num_thread uint32)

func bench_CSC(csc_func fn_CSC, alpha float64, mat_A CSC, vec_x []float64, beta float64, vec_b []float64, num_thread uint32, operation string) {
	var iter_num int = 10
	start := time.Now()
	for i := 0; i < iter_num; i++ {
		csc_func(alpha, mat_A, vec_x[:], beta, vec_b[:], num_thread)
	}
	elapsed := time.Since(start)
	avg_time := elapsed / time.Duration(iter_num)
	fmt.Println(operation, avg_time.Milliseconds())
}

type fn_CSR func(alpha float64, mat_A CSR, vec_x []float64, beta float64, vec_b []float64, num_thread uint32)

func bench_CSR(csr_func fn_CSR, alpha float64, mat_A CSR, vec_x []float64, beta float64, vec_b []float64, num_thread uint32, operation string) {
	var iter_num int = 10
	start := time.Now()
	for i := 0; i < iter_num; i++ {
		csr_func(alpha, mat_A, vec_x[:], beta, vec_b[:], num_thread)
	}
	elapsed := time.Since(start)
	avg_time := elapsed / time.Duration(iter_num)
	fmt.Println(operation, avg_time.Milliseconds())
}

func main() {
	var file_name string = "../Test_Samples/mawi_201512020030.mtx"
	var num_thread uint32 = 4

	var alpha float64 = 1
	var beta float64 = 0
	var mat_A_COO COO = mtx_load(file_name)
	var vec_x []float64 = gen_sample_vec(mat_A_COO, 1)

	var vec_b_COO_serial []float64 = gen_sample_vec(mat_A_COO, 0)
	bench_COO(coo_spmv_serial, alpha, mat_A_COO, vec_x[:], beta, vec_b_COO_serial[:], num_thread, "COO (S)       :")
	var vec_b_COO_parallel_mut_var []float64 = gen_sample_vec(mat_A_COO, 0)
	bench_COO(coo_spmv_parallel_mutex_var, alpha, mat_A_COO, vec_x[:], beta, vec_b_COO_parallel_mut_var[:], num_thread, "COO (P-M_Var) :")
	var vec_b_COO_parallel_mut_arr []float64 = gen_sample_vec(mat_A_COO, 0)
	bench_COO(coo_spmv_parallel_mutex_arr, alpha, mat_A_COO, vec_x[:], beta, vec_b_COO_parallel_mut_arr[:], num_thread, "COO (P-M_Arr) :")

	var mat_A_CSC CSC = COO_TO_CSC(mat_A_COO)
	var vec_b_CSC_serial []float64 = gen_sample_vec(mat_A_COO, 0)
	bench_CSC(csc_spmv_serial, alpha, mat_A_CSC, vec_x[:], beta, vec_b_CSC_serial[:], num_thread, "CSC (S)       :")
	var vec_b_CSC_parallel_mut_var []float64 = gen_sample_vec(mat_A_COO, 0)
	bench_CSC(csc_spmv_parallel_mutex_var, alpha, mat_A_CSC, vec_x[:], beta, vec_b_CSC_parallel_mut_var[:], num_thread, "CSC (P-M_Var) :")
	var vec_b_CSC_parallel_mut_arr []float64 = gen_sample_vec(mat_A_COO, 0)
	bench_CSC(csc_spmv_parallel_mutex_arr, alpha, mat_A_CSC, vec_x[:], beta, vec_b_CSC_parallel_mut_arr[:], num_thread, "CSC (P-M_Arr) :")

	var mat_A_CSR CSR = COO_TO_CSR(mat_A_COO)
	var vec_b_CSR_serial []float64 = gen_sample_vec(mat_A_COO, 0)
	bench_CSR(csr_spmv_serial, alpha, mat_A_CSR, vec_x[:], beta, vec_b_CSR_serial[:], num_thread, "CSR (S)       :")
	var vec_b_CSR_Parallel []float64 = gen_sample_vec(mat_A_COO, 0)
	bench_CSR(csr_spmv_parallel, alpha, mat_A_CSR, vec_x[:], beta, vec_b_CSR_Parallel[:], num_thread, "CSR (P)       :")
}
