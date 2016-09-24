#include <cstdlib>
#include <cstdio>
#include "tools.h"

float fRand(float fMin, float fMax) {
    float f = (float)rand() / RAND_MAX;
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

Timer timer = Timer();
