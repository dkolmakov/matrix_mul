#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace mul_man {
    
namespace avx {

    // Initial idea took from https://stackoverflow.com/questions/39509746/how-to-square-two-complex-doubles-with-256-bit-avx-vectors

    template <class T, std::size_t index, std::size_t step>
    struct unpack;

    template <class T>
    struct unpack<T, 0, 1> {
        static force_inline void doIt(__m256d *realB, __m256d *imagB, std::complex<T> **arr) {
            __m256d B0 = _mm256_loadu_pd((double*)(arr[0] + 0 * 4));
            __m256d B2 = _mm256_loadu_pd((double*)(arr[0] + 0 * 4 + 2));
            realB[0] = _mm256_unpacklo_pd(B0, B2);
            imagB[0] = _mm256_unpackhi_pd(B0, B2);
        }
    };

    template <class T, std::size_t index>
    struct unpack<T, index, 1> {
        static force_inline void doIt(__m256d *realB, __m256d *imagB, std::complex<T> **arr) {
            __m256d B0 = _mm256_loadu_pd((double*)(arr[0] + index * 4));
            __m256d B2 = _mm256_loadu_pd((double*)(arr[0] + index * 4 + 2));
            realB[index] = _mm256_unpacklo_pd(B0, B2);
            imagB[index] = _mm256_unpackhi_pd(B0, B2);
            unpack<T, index - 1, 1>::doIt(realB, imagB, arr);
        }
    };

    template <class T>
    struct unpack<T, 0, 2> {
        static force_inline void doIt(__m256d *realB, __m256d *imagB, std::complex<T> **arr) {
            double *to_load0 = (double*)arr[0];
            double *to_load1 = (double*)arr[1];
            realB[0] = _mm256_set_pd(to_load0[0], to_load0[2], to_load1[0], to_load1[2]);
            imagB[0] = _mm256_set_pd(to_load0[1], to_load0[3], to_load1[1], to_load1[3]);
        }
    };
    
    template <class T>
    struct unpack<T, 0, 4> {
        static force_inline void doIt(__m256d *realB, __m256d *imagB, std::complex<T> **arr) {
            double *to_load0 = (double*)arr[0];
            double *to_load1 = (double*)arr[1];
            double *to_load2 = (double*)arr[2];
            double *to_load3 = (double*)arr[3];
            realB[0] = _mm256_set_pd(to_load0[0], to_load1[0], to_load2[0], to_load3[0]);
            imagB[0] = _mm256_set_pd(to_load0[1], to_load1[1], to_load2[1], to_load3[1]);
        }
    };
    
    template <class T, std::size_t index, std::size_t step>
    struct pack;

    template <class T>
    struct pack<T, 0, 1> {
        static force_inline void doIt(__m256d *realA, __m256d *imagA, std::complex<T> **acc) {
            __m256d dst0 = _mm256_shuffle_pd(realA[0], imagA[0], 0b0000);
            __m256d dst2 = _mm256_shuffle_pd(realA[0], imagA[0], 0b1111);
            _mm256_storeu_pd((double*)(acc[0] + 0 * 4), dst0);
            _mm256_storeu_pd((double*)(acc[0] + 0 * 4 + 2), dst2);
        }
    };

    template <class T, std::size_t index>
    struct pack<T, index, 1> {
        static force_inline void doIt(__m256d *realA, __m256d *imagA, std::complex<T> **acc) {
            __m256d dst0 = _mm256_shuffle_pd(realA[index], imagA[index], 0b0000);
            __m256d dst2 = _mm256_shuffle_pd(realA[index], imagA[index], 0b1111);
            _mm256_storeu_pd((double *)(acc[0] + index * 4), dst0);
            _mm256_storeu_pd((double *)(acc[0] + index * 4 + 2), dst2);

            pack<T, index - 1, 1>::doIt(realA, imagA, acc);
        }
    };
    
    template <class T>
    struct pack<T, 0, 2> {
        static force_inline void doIt(__m256d *realA, __m256d *imagA, std::complex<T> **acc) {
            double *to_store0 = (double *)acc[0];
            double *to_store1 = (double *)acc[1];
            double *real = (double *)realA;
            double *imag = (double *)imagA;
            to_store0[0] = real[3];
            to_store0[1] = imag[3];
            to_store0[2] = real[2];
            to_store0[3] = imag[2];

            to_store1[0] = real[1];
            to_store1[1] = imag[1];
            to_store1[2] = real[0];
            to_store1[3] = imag[0];
        }
    };

    template <class T>
    struct pack<T, 0, 4> {
        static force_inline void doIt(__m256d *realA, __m256d *imagA, std::complex<T> **acc) {
            double *to_store0 = (double *)acc[0];
            double *to_store1 = (double *)acc[1];
            double *to_store2 = (double *)acc[2];
            double *to_store3 = (double *)acc[3];
            double* real = (double*)realA;
            double* imag = (double*)imagA;
            to_store0[0] = real[3];
            to_store0[1] = imag[3];
            to_store1[0] = real[2];
            to_store1[1] = imag[2];

            to_store2[0] = real[1];
            to_store2[1] = imag[1];
            to_store3[0] = real[0];
            to_store3[1] = imag[0];
        }
    };
    
    template <class T, std::size_t index>
    struct multiply;

    template <class T>
    struct multiply<T, 0> {
        static force_inline void compute(__m256d *realA, __m256d *imagA, __m256d *realB, __m256d *imagB) {
            __m256d realprod = _mm256_mul_pd(realA[0], realB[0]);
            __m256d imagprod = _mm256_mul_pd(imagA[0], imagB[0]);
            
            __m256d rAiB     = _mm256_mul_pd(realA[0], imagB[0]);
            __m256d rBiA     = _mm256_mul_pd(realB[0], imagA[0]);

            realA[0]     = _mm256_sub_pd(realprod, imagprod);
            imagA[0]     = _mm256_add_pd(rAiB, rBiA);
        }
    };

    template <class T, std::size_t index>
    struct multiply {
        static force_inline void compute(__m256d *realA, __m256d *imagA, __m256d *realB, __m256d *imagB) {
            __m256d realprod = _mm256_mul_pd(realA[index], realB[index]);
            __m256d imagprod = _mm256_mul_pd(imagA[index], imagB[index]);
            
            __m256d rAiB     = _mm256_mul_pd(realA[index], imagB[index]);
            __m256d rBiA     = _mm256_mul_pd(realB[index], imagA[index]);

            realA[index]     = _mm256_sub_pd(realprod, imagprod);
            imagA[index]     = _mm256_add_pd(rAiB, rBiA);

            multiply<T, index - 1>::compute(realA, imagA, realB, imagB);
        }
    };
    
}

}


