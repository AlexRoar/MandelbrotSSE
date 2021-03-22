template<typename T>
struct cuComplex {
    T r;
    T i;

    cuComplex(T a, T b) : r(a), i(b) {}

    T magnitude2() {
        return r * r + i * i;
    }

    cuComplex operator*(const cuComplex &a) {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    cuComplex operator+(const cuComplex &a) {
        return cuComplex(r + a.r, i + a.i);
    }

    T abs() const {
        return sqrt(magnitude2());
    }
};