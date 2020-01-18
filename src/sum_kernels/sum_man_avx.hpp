#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace sum_man {
    
namespace avx {

    template <class T, std::size_t index, std::size_t step>
    struct unpack;

    template <class T>
    struct unpack<T, 0, 1> {
        static force_inline void doIt(__m256d *vals, std::complex<T> **arr) {
            vals[0] = _mm256_loadu_pd((double*)(arr[0] + 0 * 2));
        }
    };

    template <class T, std::size_t index>
    struct unpack<T, index, 1> {
        static force_inline void doIt(__m256d *vals, std::complex<T> **arr) {
            vals[index] = _mm256_loadu_pd((double*)(arr[0] + index * 2));
            unpack<T, index - 1, 1>::doIt(vals, arr);
        }
    };

    template <class T>
    struct unpack<T, 0, 2> {
        static force_inline void doIt(__m256d *vals, std::complex<T> **arr) {
            double *to_load0 = (double*)arr[0];
            double *to_load1 = (double*)arr[1];
            vals[0] = _mm256_setr_pd(to_load0[0], to_load0[1], to_load1[0], to_load1[1]);
        }
    };

    template <class T, std::size_t index, std::size_t step>
    struct pack;

    template <class T>
    struct pack<T, 0, 1> {
        static force_inline void doIt(std::complex<T> **dst, __m256d *acc) {
            _mm256_storeu_pd((double*)(dst[0] + 0 * 2), acc[0]);
        }
    };

    template <class T, std::size_t index>
    struct pack<T, index, 1> {
        static force_inline void doIt(std::complex<T> **dst, __m256d *acc) {
            _mm256_storeu_pd((double*)(dst[0] + index * 2), acc[index]);
            pack<T, index - 1, 1>::doIt(dst, acc);
        }
    };

    template <class T>
    struct pack<T, 0, 2> {
        static force_inline void doIt(std::complex<T> **dst, __m256d *acc) {
            double *to_store0 = (double *)dst[0];
            double *to_store1 = (double *)dst[1];
            double* result = (double*)acc;
            to_store0[0] = result[0];
            to_store0[1] = result[1];
            to_store1[0] = result[2];
            to_store1[1] = result[3];
        }
    };
    
    template <class T, std::size_t index>
    struct summation;
    
    template <class T>
    struct summation<T, 0> {
        static force_inline void doIt(__m256d *acc, __m256d *vals) {
            acc[0] = _mm256_add_pd(acc[0], vals[0]);
        }
    };
    
    template <class T, std::size_t index>
    struct summation {
        static force_inline void doIt(__m256d *acc, __m256d *vals) {
            acc[index] = _mm256_add_pd(acc[index], vals[index]);

            summation<T, index - 1>::doIt(acc, vals);
        }
    };
    
}
}


