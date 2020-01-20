#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <array>
#include <vector>

#include "mm.hpp"

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
