OUTPUT_DIR = output/
SRC_MAIN_DIR = src_mains/

CC = g++
CFLAGS = -Wall -Wextra -Wconversion -Wpedantic -Wno-unused-result -Wno-unused-parameter -std=c++20 -O2
 
demo_dyn:
	$(CC) $(SRC_MAIN_DIR)demo_dynamic.cpp -o $(OUTPUT_DIR)demo_dynamic.out $(CFLAGS) -fopenmp

demo_sta:
	$(CC) $(SRC_MAIN_DIR)demo_static.cpp -o $(OUTPUT_DIR)demo_static.out $(CFLAGS) -fopenmp

bench_pThread:
	$(CC) $(SRC_MAIN_DIR)test_pThread_Final.cpp -o $(OUTPUT_DIR)test_pthread.out $(CFLAGS) -lpthread

bench_stdThread:
	$(CC) $(SRC_MAIN_DIR)test_stdthread_Final.cpp -o $(OUTPUT_DIR)test_stdthread.out $(CFLAGS) -lpthread

bench_openMP:
	$(CC) $(SRC_MAIN_DIR)test_openmp_Final.cpp -o $(OUTPUT_DIR)test_openmp.out $(CFLAGS) -fopenmp

bench_openMP_ADV:
	$(CC) $(SRC_MAIN_DIR)test_openmp_Advanced.cpp -o $(OUTPUT_DIR)test_openmp_Advanced.out $(CFLAGS) -fopenmp

bench_struct_diff:
	$(CC) $(SRC_MAIN_DIR)test_struct_diff.cpp -o $(OUTPUT_DIR)test_struct_diff.out $(CFLAGS) -fopenmp

all: demo_dyn demo_sta bench_pThread bench_stdThread bench_openMP bench_openMP_ADV bench_struct_diff
