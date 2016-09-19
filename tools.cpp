#include <cstdlib>
#include <cstdio>
#include "tools.h"

double fRand(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

void Timer::start() {
    this->start_time = clock();
}

float Timer::stop(const char header[]) {
    float total = ((float)(clock() - this->start_time)) / CLOCKS_PER_SEC;
    if (header != NULL) {
        printf("%s: %f\n", header, total);
    }
    return total;

}

extern Timer timer = Timer();
