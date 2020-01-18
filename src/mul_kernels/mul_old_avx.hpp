#pragma once

#include <immintrin.h>
#include <complex>

#include "common.hpp"

namespace mul_old {
    
namespace avx {

    template <class T, std::size_t index, std::size_t step>
    struct unpack;

    template <class T>
    struct unpack<T, 0, 1> {
        static force_inline void doIt(__m256d *vals, std::complex<T> **arr) {
            vals[0] = _mm256_loadu_pd((double*)(arr[0] + 0));
        }
    };

    template <class T, std::size_t index>
    struct unpack<T, index, 1> {
        static force_inline void doIt(__m256d *vals, std::complex<T> **arr) {
            vals[index] = _mm256_loadu_pd((double*)(arr[0] + 2 * index));
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
            _mm256_storeu_pd((double*)(dst[0] + 2 * 0), acc[0]);
        }
    };

    template <class T, std::size_t index>
    struct pack<T, index, 1> {
        static force_inline void doIt(std::complex<T> **dst, __m256d *acc) {
            _mm256_storeu_pd((double*)(dst[0] + 2 * index), acc[index]);
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

//     template <class T>
//     struct pack<T, 0, 2> {
//         static force_inline void doIt(std::complex<T> **dst, __m256d *acc) {
//             std::complex<T> to_store[2];
//             _mm256_storeu_pd((double*)(to_store), acc[0]);
//             *(dst[0]) = to_store[0];
//             *(dst[1]) = to_store[1];
//         }
//     };

    template <class T, std::size_t index>
    struct multiply;
    
    template <class T>
    struct multiply<T, 0> {
//         constexpr static __m256d neg = {1.0, -1.0, 1.0, -1.0};
        
        static force_inline void doIt(__m256d *acc, __m256d *vals) {
            /* Step 1: Multiply res and vec2 */
            __m256d vec0 = _mm256_mul_pd(acc[0], vals[0]);

            /* Step 2: Switch the real and imaginary elements of vec2 */
            vals[0] = _mm256_permute_pd(vals[0], 0x5); 
            
            /* Step 3: Negate the imaginary elements of vec2 */
//             __m256d neg = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);
//             vals[0] = _mm256_mul_pd(vals[0], neg);       // this is much slower than XOR
            // Flipping the sign bit in vec1 lets this run in parallel with the shuffle on vec2, reducing latency
            //__m256d odd_signbits = _mm256_castsi256_pd( _mm256_setr_epi64x(0, 1ULL<<63, 0, 1ULL<<63));
            __m256d odd_signbits = _mm256_setr_pd(0, -0.0, 0, -0.0);
            acc[0] = _mm256_xor_pd(acc[0], odd_signbits);
            
            /* Step 4: Multiply vec1 and the modified vec2 */
            __m256d vec1 = _mm256_mul_pd(acc[0], vals[0]);

            /* Horizontally subtract the elements in vec3 and vec4 */
            acc[0] = _mm256_hsub_pd(vec0, vec1);
        }
    };
    
    // Original version id from https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX

    template <class T, std::size_t index>
    struct multiply {
        static force_inline void doIt(__m256d *acc, __m256d *vals) {
            /* Step 1: Multiply res and vec2 */
            __m256d vec0 = _mm256_mul_pd(acc[index], vals[index]);

            /* Step 2: Switch the real and imaginary elements of vec2 */
            vals[index] = _mm256_permute_pd(vals[index], 0x5); 
            
            /* Step 3: Negate the imaginary elements of vec2 */
                //  vec2 = _mm256_mul_pd(vec2, neg);       // this is much slower than XOR
            // Flipping the sign bit in vec1 lets this run in parallel with the shuffle on vec2, reducing latency
            //__m256d odd_signbits = _mm256_castsi256_pd( _mm256_setr_epi64x(0, 1ULL<<63, 0, 1ULL<<63));
            __m256d odd_signbits = _mm256_setr_pd(0, -0.0, 0, -0.0);
            acc[index] = _mm256_xor_pd(acc[index], odd_signbits);
            
            /* Step 4: Multiply vec1 and the modified vec2 */
            __m256d vec1 = _mm256_mul_pd(acc[index], vals[index]);

            /* Horizontally subtract the elements in vec3 and vec4 */
            acc[index] = _mm256_hsub_pd(vec0, vec1);

            multiply<T, index - 1>::doIt(acc, vals);
        }
    };
    
    
}
}


