#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace sum_man {
    
namespace sse {

    template <class T, std::size_t index, std::size_t step>
    struct unpack;

    template <class T>
    struct unpack<T, 0, 1> {
        static force_inline void doIt(__m128d *vals, std::complex<T> **arr) {
            vals[0] = _mm_loadu_pd((double*)(arr[0] + 0));
        }
    };

    template <class T, std::size_t index>
    struct unpack<T, index, 1> {
        static force_inline void doIt(__m128d *vals, std::complex<T> **arr) {
            vals[index] = _mm_loadu_pd((double*)(arr[0] + index));
            unpack<T, index - 1, 1>::doIt(vals, arr);
        }
    };

    template <class T, std::size_t index, std::size_t step>
    struct pack;

    template <class T>
    struct pack<T, 0, 1> {
        static force_inline void doIt(std::complex<T> **dst, __m128d *acc) {
            _mm_storeu_pd((double*)(dst[0] +  0), acc[0]);
        }
    };

    template <class T, std::size_t index>
    struct pack<T, index, 1> {
        static force_inline void doIt(std::complex<T> **dst, __m128d *acc) {
            _mm_storeu_pd((double*)(dst[0] + index), acc[index]);
            pack<T, index - 1, 1>::doIt(dst, acc);
        }
    };

    template <class T, std::size_t index>
    struct summation;
    
    template <class T>
    struct summation<T, 0> {
        static force_inline void doIt(__m128d *acc, __m128d *vals) {
            acc[0] = _mm_add_pd(acc[0], vals[0]);
        }
    };
    
    template <class T, std::size_t index>
    struct summation {
        static force_inline void doIt(__m128d *acc, __m128d *vals) {
            acc[index] = _mm_add_pd(acc[index], vals[index]);

            summation<T, index - 1>::doIt(acc, vals);
        }
    };
    
}
}


