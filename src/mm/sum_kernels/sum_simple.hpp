#pragma once

#include <complex>

#include "common.hpp"

namespace sum_simple {
    
    struct chunk_sum {
        static std::string get_label() {
            return "sum_simple";
        }
        
        template <class T, std::size_t chunk_size, std::size_t n_chunks>
        struct core {
            static force_inline void compute(std::complex<T> **acc, std::complex<T> **left, std::complex<T> **right) {
                for (std::size_t i = 0; i < n_chunks; i++) {
                    for (std::size_t j = 0; j < chunk_size; j++)
                        acc[i][j] = left[i][j] + right[i][j];
                }
            }
        };
    };
    
}


