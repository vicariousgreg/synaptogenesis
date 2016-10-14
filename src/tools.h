#ifndef tools_h
#define tools_h

#include <cstdlib>
#include <cstdio>
#include <ctime>

/* Calculates a random float between |fMin| and |fMax| */
inline float fRand(float fMin, float fMax) {
    float f = (float)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

/* Timer class.
 * Can be used to keep track of runtimes.
 * Call start() to set a start time.
 * Call stop() to calculate time difference since last start() call.
 */
class Timer {
    public:
        /* Sets a start time */
        void start() {
            this->start_time = clock();
        }

        /* Calculates elapsed time since last start() call.
         * If |header| is provided, the time will be printed.
         */
        float query(const char header[]) {
            float total = ((float)(clock() - this->start_time)) / CLOCKS_PER_SEC;
            if (header != NULL) {
                printf("%s: %f\n", header, total);
            }
            return total;
        }

        /* Waits until the duration exceeds the given limit */
        void wait(float limit) {
            float total;
            do {
                total = ((float)(clock() - this->start_time)) / CLOCKS_PER_SEC;
            } while (total < limit);
        }
        
    private:
        // Timestamp from last start call()
        clock_t start_time;
};

#endif
