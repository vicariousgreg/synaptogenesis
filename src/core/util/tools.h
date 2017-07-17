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
static std::uniform_real_distribution<double> f_distribution(0.0, 1.0);
static std::uniform_int_distribution<int> i_distribution(0,9);

/* Random float functions */
inline float fRand() {
    return f_distribution(generator);
}
inline float fRand(float fMax) {
    return f_distribution(generator) * fMax;
}
inline float fRand(float fMin, float fMax) {
    return fMin + (f_distribution(generator) * (fMax - fMin));
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
            float total;
            Time_point curr_time;
            do {
                curr_time = CClock::now();
                total = get_diff(curr_time, this->start_time);
            } while (total < limit);
        }

    private:
        // Timestamp from last start call()
        Time_point start_time = CClock::now();
};

#endif
