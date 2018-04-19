#include "util/tools.h"

/* Static float function */
float fSet(float* arr, int size, float val, float fraction) {
    if (fraction == 1.0) {
        for (int i = 0 ; i < size ; ++i) arr[i] = val;
    } else {
        auto dist = std::uniform_real_distribution<float>(0.0, 1.0);
        for (int i = 0 ; i < size ; ++i)
            if (dist(generator) < fraction) arr[i] = val;
    }
}

void fRand(float* arr, int size, float fMin, float fMax, float fraction) {
    auto dist = std::uniform_real_distribution<float>(fMin, fMax);
    if (fraction == 1.0) {
        for (int i = 0 ; i < size ; ++i) arr[i] = dist(generator);
    } else {
        auto f_dist = std::uniform_real_distribution<float>(0.0, 1.0);
        for (int i = 0 ; i < size ; ++i)
            if (f_dist(generator) < fraction)
                arr[i] = dist(generator);
    }
}

void iRand(int* arr, int size, int iMin, int iMax, float fraction) {
    auto dist = std::uniform_int_distribution<int>(iMin,iMax);
    if (fraction == 1.0)
        for (int i = 0 ; i < size ; ++i) arr[i] = dist(generator);
    else {
        auto f_dist = std::uniform_real_distribution<float>(0.0, 1.0);
        for (int i = 0 ; i < size ; ++i)
            if (f_dist(generator) < fraction)
                arr[i] = dist(generator);
    }
}
