//
// Created by Александр Дремов on 19.03.2021.
//
#include <cmath>
#include "Complex.cuh"

template<typename c_type>
__device__ __host__ void Complex<c_type>::init(c_type rN, c_type iN) {
    r = rN;
    i = iN;
}


template<typename c_type>
__device__ __host__ void Complex<c_type>::init() {
    r = 0;
    i = 0;
}

template<typename c_type>
__device__ __host__ void Complex<c_type>::dest() {}


template<typename c_type>
__device__ __host__ void Complex<c_type>::Delete() {
    dest();
    free(this);
}

template<typename c_type>
__device__ __host__ void Complex<c_type>::add(const Complex &other) {
    r += other.r;
    i += other.i;
}

template<typename c_type>
__device__ __host__ void Complex<c_type>::sub(const Complex &other) {
    r -= other.r;
    i -= other.i;
}

template<typename c_type>
__device__ __host__ c_type Complex<c_type>::abs() const {
    return sqrt(r * r + i * i);
}

template<typename c_type>
__device__ __host__ c_type Complex<c_type>::absNoSqrt() const {
    return r * r + i * i;
}

template<typename c_type>
__device__ __host__ void Complex<c_type>::mul(const Complex &other) {
    const c_type lastR = r;
    r = r * other.r - i * other.i;
    i = i * other.r + other.i + lastR;
}

template<typename c_type>
__device__ __host__ void Complex<c_type>::square() {
    const c_type lastR = r;
    r = r * r - i * i;
    i = 2 * i * lastR;
}

template<typename c_type>
__device__ __host__ Complex<c_type> Complex<c_type>::squared() {
    Complex n = *this;
    n.square();
    return n;
}


template<typename c_type>
__device__ __host__ bool Complex<c_type>::isNan() {
    return r == NAN || i == NAN;
}
