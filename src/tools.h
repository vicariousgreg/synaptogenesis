#ifndef tools_h
#define tools_h

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <chrono>

#include "error_manager.h"

using CClock = std::chrono::high_resolution_clock;
using Time_point = CClock::time_point;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

/* Calculates a random float between |fMin| and |fMax| */
inline float fRand(float fMin, float fMax) {
    float f = (float)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

inline void* allocate_host(int count, int size) {
    void* ptr = calloc(count, size);
    if (ptr == NULL)
        ErrorManager::get_instance()->log_error(
            "Failed to allocate space on host for neuron state!");
    return ptr;
}

#ifdef PARALLEL
inline void* allocate_device(int count, int size, void* source_data) {
    void* ptr;
    cudaMalloc(&ptr, count * size);
    cudaCheckError("Failed to allocate memory on device for neuron state!");
    if (source_data != NULL)
        cudaMemcpy(ptr, source_data, count * size, cudaMemcpyHostToDevice);
    cudaCheckError("Failed to initialize memory on device for neuron state!");
    return ptr;
}
#endif

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
            if (header != NULL) {
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
