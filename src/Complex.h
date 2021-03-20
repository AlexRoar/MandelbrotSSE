//
// Created by Александр Дремов on 19.03.2021.
//

#ifndef MANDELBROT_COMPLEX_H
#define MANDELBROT_COMPLEX_H

#include <cstdlib>
#include <cmath>

struct Complex {
    typedef double c_type;
    c_type r;
    c_type i;

    void init(c_type rN, c_type iN){
        r = rN;
        i = iN;
    }

    void init(){
        r = 0;
        i = 0;
    }

    void dest(){}

    static Complex* New(){
        Complex* thou = static_cast<Complex*>(calloc(1, sizeof(Complex)));
        thou->init();
        return thou;
    }

    void Delete(){
        dest();
        free(this);
    }

    void add(const Complex& other) {
        r += other.r;
        i += other.i;
    }

    void sub(const Complex& other) {
        r -= other.r;
        i -= other.i;
    }

    [[nodiscard]] c_type abs() const {
        return sqrt(r * r + i * i);
    }

    [[nodiscard]] c_type absNoSqrt() const {
        return r * r + i * i;
    }

    void mul(const Complex& other) {
        const c_type lastR = r;
        r = r * other.r - i * other.i;
        i = i * other.r + other.i + lastR;
    }

    void square() {
        const c_type lastR = r;
        r = r * r - i * i;
        i = 2 * i * lastR;
    }

    Complex squared(){
        Complex n = *this;
        n.square();
        return n;
    }

    bool isNan(){
        return r == NAN || i == NAN;
    }
};

#endif //MANDELBROT_COMPLEX_H
