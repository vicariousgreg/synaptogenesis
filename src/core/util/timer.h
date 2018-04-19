#ifndef timer_h
#define timer_h

#include <ctime>
#include <chrono>

// Timer namespaces
using CClock = std::chrono::high_resolution_clock;
using TimePoint = CClock::time_point;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

static inline float get_diff(TimePoint a, TimePoint b)
    { return (float)duration_cast<milliseconds>(a - b).count() / 1000; }

/* Timer class.
 * Can be used to keep track of runtimes.
 * Call start() to set a start time.
 * Call stop() to calculate time difference since last start() call.
 */
class Timer {
    public:
        /* Sets a start time */
        void reset() { start_time = CClock::now(); }

        /* Calculates elapsed time since last start() call.
         * If |header| is provided, the time will be printed.
         */
        float query(const char header[]) {
            float total = get_diff(CClock::now(), this->start_time);
            if (header != nullptr)
                printf("%s: %f\n", header, total);
            return total;
        }

        /* Waits until the duration exceeds the given limit */
        void wait(float limit)
            { while (get_diff(CClock::now(), this->start_time) < limit); }

    private:
        // Timestamp from last start call()
        TimePoint start_time = CClock::now();
};

#endif
