#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"
#include "mul_simple.hpp"
#include "mul_avx.hpp"
#include "mul_sse.hpp"

namespace mul_man {
    
namespace impl {

#ifdef __AVX512F__
    constexpr std::size_t VALS_PER_OP = 8;
#elif __AVX__
    constexpr std::size_t VALS_PER_OP = 4;
#elif __SSE4_1__
    constexpr std::size_t VALS_PER_OP = 2;
#else
    constexpr std::size_t VALS_PER_OP = 1;
#endif
    
    template <class T, std::size_t chunk_size, std::size_t step, std::size_t parity_checker = (chunk_size * step) % VALS_PER_OP, std::size_t reg_size = av::SIMD_REG_SIZE>
    struct chunk_mul;

    template <class T, std::size_t chunk_size, std::size_t step>
    struct chunk_mul<T, chunk_size, step, 0, 32> {
        constexpr static std::size_t portion_size = chunk_size * step;
        constexpr static std::size_t vectors_number = portion_size / VALS_PER_OP;
        
        static force_inline void compute(std::complex<T> **acc, std::complex<T> **left, std::complex<T> **right) {
            __m256d realA[vectors_number];
            __m256d imagA[vectors_number]; 
            avx::unpack<T, vectors_number - 1, step>::doIt(realA, imagA, left);
            
            __m256d realB[vectors_number];
            __m256d imagB[vectors_number];
            avx::unpack<T, vectors_number - 1, step>::doIt(realB, imagB, right);

            avx::multiply<T, vectors_number - 1>::compute(realA, imagA, realB, imagB);
            avx::pack<T, vectors_number - 1, step>::doIt(realA, imagA, acc);
        }
    };

    template <class T, std::size_t chunk_size, std::size_t step>
    struct chunk_mul<T, chunk_size, step, 0, 16> {
        constexpr static std::size_t portion_size = chunk_size * step;
        constexpr static std::size_t vectors_number = portion_size / VALS_PER_OP;
        
        static force_inline void compute(std::complex<T> **acc, std::complex<T> **left, std::complex<T> **right) {
            __m128d realA[vectors_number];
            __m128d imagA[vectors_number]; 
            sse::unpack<T, vectors_number - 1, step>::doIt(realA, imagA, left);
            
            __m128d realB[vectors_number];
            __m128d imagB[vectors_number];
            sse::unpack<T, vectors_number - 1, step>::doIt(realB, imagB, right);

            sse::multiply<T, vectors_number - 1>::compute(realA, imagA, realB, imagB);
            sse::pack<T, vectors_number - 1, step>::doIt(realA, imagA, acc);
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t step, std::size_t parity_checker, std::size_t reg_size>
    struct chunk_mul {
        static force_inline void compute(std::complex<T> **acc, std::complex<T> **left, std::complex<T> **right) {
            mul_simple::chunk_mul::core<T, chunk_size, step>::compute(acc, left, right);
        }
    };
    
    
    template <class T, std::size_t chunk_size, std::size_t step, std::size_t index>
    struct unroll_chunks;

    template <class T, std::size_t chunk_size, std::size_t step>
    struct unroll_chunks<T, chunk_size, step, 0> {
        static force_inline void compute(std::complex<T> **acc, std::complex<T> **left, std::complex<T> **right) {
            chunk_mul<T, chunk_size, step>::compute(&acc[0], &left[0], &right[0]);
        }
    };
    
    template <class T, std::size_t chunk_size, std::size_t step, std::size_t index>
    struct unroll_chunks {
        static force_inline void compute(std::complex<T> **acc, std::complex<T> **left, std::complex<T> **right) {
            unroll_chunks<T, chunk_size, step, index - step>::compute(acc, left, right);
            chunk_mul<T, chunk_size, step>::compute(&acc[index], &left[index], &right[index]);
        }
    };
    
}

    struct chunk_mul {
        static std::string get_label() {
            return "mul_man";
        }
        
        template <class T, std::size_t chunk_size, std::size_t n_chunks>
        struct core {
            constexpr static std::size_t step = (chunk_size < impl::VALS_PER_OP && (impl::VALS_PER_OP % chunk_size == 0) && (n_chunks % (impl::VALS_PER_OP / chunk_size)  == 0)) ? impl::VALS_PER_OP / chunk_size : 1;

            static force_inline void compute(std::complex<T> **acc, std::complex<T> **left, std::complex<T> **right) {
                impl::unroll_chunks<T, chunk_size, step, n_chunks - step>::compute(acc, left, right);
            }
        };
    };
}


