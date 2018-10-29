#include "util/tools.h"

float fSet(float* arr, int size, float val, float fraction) {
    if (fraction == 1.0) {
        _Pragma("omp parallel for")
        for (int i = 0 ; i < size ; ++i) arr[i] = val;
    } else {
        auto dist = std::uniform_real_distribution<float>(0.0, 1.0);
        _Pragma("omp parallel for")
        for (int i = 0 ; i < size ; ++i)
            if (dist(generator) < fraction) arr[i] = val;
    }
}

void fClear(float* arr, int size) {
    fSet(arr, size, 0.0);
}

void fRand(float* arr, int size, float fMin, float fMax, float fraction) {
    auto dist = std::uniform_real_distribution<float>(fMin, fMax);
    if (fraction == 1.0) {
        _Pragma("omp parallel for")
        for (int i = 0 ; i < size ; ++i) arr[i] = dist(generator);
    } else {
        auto f_dist = std::uniform_real_distribution<float>(0.0, 1.0);
        _Pragma("omp parallel for")
        for (int i = 0 ; i < size ; ++i)
            if (f_dist(generator) < fraction)
                arr[i] = dist(generator);
    }
}

void iRand(int* arr, int size, int iMin, int iMax, float fraction) {
    auto dist = std::uniform_int_distribution<int>(iMin,iMax);
    if (fraction == 1.0)
        _Pragma("omp parallel for")
        for (int i = 0 ; i < size ; ++i) arr[i] = dist(generator);
    else {
        auto f_dist = std::uniform_real_distribution<float>(0.0, 1.0);
        _Pragma("omp parallel for")
        for (int i = 0 ; i < size ; ++i)
            if (f_dist(generator) < fraction)
                arr[i] = dist(generator);
    }
}



void fRand_gaussian(float* arr, int size,
        float mean, float std_dev, float max, float fraction) {
    std::normal_distribution<double> dist(mean, std_dev);

    if (fraction == 1.0) {
        _Pragma("omp parallel for")
        for (int i = 0 ; i < size ; ++i)
            arr[i] = std::min((double)max, std::max(0.0, dist(generator)));
    } else {
        std::uniform_real_distribution<double> f_dist(0.0, 1.0);
        _Pragma("omp parallel for")
        for (int i = 0 ; i < size ; ++i)
            arr[i] = (f_dist(generator) < fraction)
                ? std::min((double)max, std::max(0.0, dist(generator)))
                : 0.0;
    }
}

void fRand_lognormal(float* arr, int size,
        float mean, float std_dev, float max, float fraction) {
    std::lognormal_distribution<double> dist(mean, std_dev);

    if (fraction == 1.0) {
        _Pragma("omp parallel for")
        for (int i = 0 ; i < size ; ++i)
            arr[i] = std::min((double)max, std::max(0.0, dist(generator)));
    } else {
        std::uniform_real_distribution<double> f_dist(0.0, 1.0);
        _Pragma("omp parallel for")
        for (int i = 0 ; i < size ; ++i)
            arr[i] = (f_dist(generator) < fraction)
                ? std::min((double)max, std::max(0.0, dist(generator)))
                : 0.0;
    }
}

void fRand_powerlaw(float* arr, int size,
        float exponent, float min, float max, float fraction) {
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    float coeff_a = pow(max, 1.0-exponent);
    float coeff_b = pow(std::max(min, 0.00001f), 1.0-exponent);
    float coeff = coeff_a - coeff_b;
    float pow_exp = 1.0 / (1.0-exponent);

    if (fraction == 1.0) {
        _Pragma("omp parallel for")
        for (int i = 0 ; i < size ; ++i)
            arr[i] = pow(coeff * dist(generator) + coeff_b, pow_exp);
    } else {
        _Pragma("omp parallel for")
        for (int i = 0 ; i < size ; ++i)
            arr[i] = (dist(generator) < fraction)
                ? pow(coeff * dist(generator) + coeff_b, pow_exp)
                : 0.0;
    }
}

/* Clears the diagonal of a weight matrix */
void clear_diagonal(float *mat, int dim) {
    _Pragma("omp parallel for")
    for (int i = 0 ; i < dim ; ++i)
        mat[i * dim + i] = 0.0;
}

