#ifndef tools_h
#define tools_h

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <climits>
#include <chrono>
#include <random>

#include "util/error_manager.h"

using CClock = std::chrono::high_resolution_clock;
using Time_point = CClock::time_point;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

static std::default_random_engine generator(time(0));

/* Static float function */
inline float fSet(float* arr, int size, float val, float fraction=1.0) {
    if (fraction == 1.0) {
        for (int i = 0 ; i < size ; ++i) arr[i] = val;
    } else {
        auto dist = std::uniform_real_distribution<float>(0.0, 1.0);
        for (int i = 0 ; i < size ; ++i)
            if (dist(generator) < fraction) arr[i] = val;
    }
}

/* Random float functions */
inline float fRand() {
    return std::uniform_real_distribution<float>(0.0, 1.0)(generator);
}
inline float fRand(float fMax) {
    return std::uniform_real_distribution<float>(0.0, fMax)(generator);
}
inline float fRand(float fMin, float fMax) {
    return std::uniform_real_distribution<float>(fMin, fMax)(generator);
}
inline void fRand(float* arr, int size, float fMin, float fMax, float fraction=1.0) {
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

/* Random int functions */
inline int iRand() {
    return std::uniform_int_distribution<int>(0,INT_MAX)(generator);
}
inline int iRand(int iMax) {
    return std::uniform_int_distribution<int>(0,iMax)(generator);
}
inline int iRand(int iMin, int iMax) {
    return std::uniform_int_distribution<int>(iMin,iMax)(generator);
}
inline void iRand(int* arr, int size, int iMin, int iMax, float fraction=1.0) {
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


static float get_diff(Time_point a, Time_point b) {
    return (float)duration_cast<milliseconds>(a - b).count() / 1000;
}

/* Timer class.
 * Can be used to keep track of runtimes.
 * Call start() to set a start time.
 * Call stop() to calculate time difference since last start() call.
 */
class Timer {
    public:
        /* Sets a start time */
        void reset() {
            start_time = CClock::now();
        }

        /* Calculates elapsed time since last start() call.
         * If |header| is provided, the time will be printed.
         */
        float query(const char header[]) {
            Time_point curr_time = CClock::now();
            float total = get_diff(curr_time, this->start_time);
            if (header != nullptr) {
                printf("%s: %f\n", header, total);
            }
            return total;
        }

        /* Waits until the duration exceeds the given limit */
        void wait(float limit) {
            while (get_diff(CClock::now(), this->start_time) < limit);
        }

    private:
        // Timestamp from last start call()
        Time_point start_time = CClock::now();
};

#endif
