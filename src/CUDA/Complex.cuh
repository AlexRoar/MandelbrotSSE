//
// Created by Александр Дремов on 19.03.2021.
//

#ifndef COMPLEXCUDA
#define COMPLEXCUDA


#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <helper_cuda.h>
#include "../../../../../usr/local/cuda/include/cuda_runtime.h"

template<typename c_type>
struct Complex {
    c_type r;
    c_type i;

    __host__
    __device__ void init(c_type rN, c_type iN);

    __host__
    __device__ void init();

    __host__
    __device__ void dest();

    __host__
    __device__ static Complex* New(){
        auto *thou = static_cast<Complex<c_type> *>(calloc(1, sizeof(Complex<c_type>)));
        thou->init();
        return thou;
    }

    __host__
    __device__ void Delete();

    __host__
    __device__ void add(const Complex& other);

    __host__
    __device__ void sub(const Complex& other);

    __host__
    __device__ c_type abs() const;

    __host__
    __device__ c_type absNoSqrt() const;

    __host__
    __device__ void mul(const Complex& other);

    __host__
    __device__ void square();

    __host__
    __device__ Complex squared();

    __host__
    __device__ bool isNan();
};
#endif