#ifndef tools_h
#define tools_h

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <random>

#include "util/error_manager.h"

using CClock = std::chrono::high_resolution_clock;
using Time_point = CClock::time_point;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

static std::default_random_engine generator(time(0));
static std::uniform_real_distribution<double> distribution(0.0, 1.0);

/* Random float functions */
inline float fRand() {
    return distribution(generator);
}
inline float fRand(float fMax) {
    return distribution(generator) * fMax;
}
inline float fRand(float fMin, float fMax) {
    return fMin + (distribution(generator) * (fMax - fMin));
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
