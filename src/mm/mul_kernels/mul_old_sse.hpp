#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace mul_old {
    
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
    struct multiply;
    
    template <class T>
    struct multiply<T, 0> {
        static force_inline void doIt(__m128d *acc, __m128d *vals) {
            __m128d tmp0 = _mm_mul_pd(acc[0], vals[0]);
            __m128d tmp1 = _mm_shuffle_pd(vals[0], vals[0], 1);

            __m128d odd_signbits = _mm_setr_pd(0, -0.0);
            acc[0] = _mm_xor_pd(acc[0], odd_signbits);

            __m128d tmp2 = _mm_mul_pd(acc[0], tmp1);

            acc[0] = _mm_hsub_pd(tmp0, tmp2);
        }
    };
    
    template <class T, std::size_t index>
    struct multiply {
        static force_inline void doIt(__m128d *acc, __m128d *vals) {
            __m128d tmp0 = _mm_mul_pd(acc[index], vals[index]);
            __m128d tmp1 = _mm_shuffle_pd(vals[index], vals[index], 1);

            __m128d odd_signbits = _mm_setr_pd(0, -0.0);
            acc[index] = _mm_xor_pd(acc[index], odd_signbits);
            
            __m128d tmp2 = _mm_mul_pd(acc[index], tmp1);
            
            acc[index] = _mm_hsub_pd(tmp0, tmp2);

            multiply<T, index - 1>::doIt(acc, vals);
        }
    };
    
}

}


