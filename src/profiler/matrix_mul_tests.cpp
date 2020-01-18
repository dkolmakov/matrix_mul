#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <array>
#include <vector>

#include "mul_simple.hpp"
#include "mul_unroll.hpp"
#include "mul_old.hpp"
#include "mul_man.hpp"

#include "sum_simple.hpp"
#include "sum_unroll.hpp"
#include "sum_man.hpp"

#include "matrix_mul.hpp"

#include "test_harness.hpp"

using namespace av;

typedef KernelParameters<std::size_t, 1, 2, 4, 8, 16, 32> chunk_sizes;
typedef KernelParameters<std::size_t, 1, 2, 4, 8> chunk_numbers;
typedef Kernels<mul_simple::chunk_mul, 
                mul_unroll::chunk_mul, 
                mul_man::chunk_mul, 
                mul_old::chunk_mul> mul_kernels;
typedef Kernels<sum_simple::chunk_sum, 
                sum_unroll::chunk_sum, 
                sum_man::chunk_sum> sum_kernels;
typedef Combinations<mul_kernels, sum_kernels, chunk_sizes, chunk_numbers> tuples;

typedef TestHarness<matrix_mul::test_function<double>, tuples::val> matrix_mul_harness;
typedef matrix_mul::test_function<double>::input_data matrix_mul_input;


int main(int argc, char **argv) {
    if (argc < 3)
        return 1;
    
    std::size_t count = atoi(argv[1]);
    std::size_t repeats = atoi(argv[2]);

    Benchmark<matrix_mul_input>* matrix_mul_benchmark = matrix_mul_harness::prepare_benchmark("matrix_mul");
    
    std::cout << av::inst_set << " instruction set" << std::endl;
    
    for (std::size_t i = 0; i < repeats; i++)
        matrix_mul_benchmark->run(count);
    
    matrix_mul_benchmark->print_results();

    delete matrix_mul_benchmark;
    return 0;
}
