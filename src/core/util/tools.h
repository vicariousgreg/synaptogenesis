#ifndef tools_h
#define tools_h

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <climits>
#include <chrono>
#include <random>

#include <cmath>

// Different min, max, and assert functions are used on the host and device
#ifdef __CUDACC__
#define MIN min
#define MAX max
#else
#include <algorithm>
#include <assert.h>
#define MIN std::fmin
#define MAX std::fmax
#endif

// Random number generator
static std::default_random_engine generator(std::random_device{}());

// Float array setter
float fSet(float* arr, int size, float val, float fraction=1.0);

/* Float and integer random generators
 *   Defaults
 *   0 - Max
 *   Min - Max
 */
inline float fRand()
    { return std::uniform_real_distribution<float>(0.0, 1.0)(generator); }
inline float fRand(float fMax)
    { return std::uniform_real_distribution<float>(0.0, fMax)(generator); }
inline float fRand(float fMin, float fMax)
    { return std::uniform_real_distribution<float>(fMin, fMax)(generator); }
inline int iRand()
    { return std::uniform_int_distribution<int>(0,INT_MAX)(generator); }
inline int iRand(int iMax)
    { return std::uniform_int_distribution<int>(0,iMax)(generator); }
inline int iRand(int iMin, int iMax)
    { return std::uniform_int_distribution<int>(iMin,iMax)(generator); }

/* Flat and integer array random generators */
void fRand(float* arr, int size, float fMin, float fMax, float fraction=1.0);
void iRand(int* arr, int size, int iMin, int iMax, float fraction=1.0);


/* Transposes a matrix in place */
template <typename T>
void transpose_matrix_in_place(T* data, int original_rows, int original_cols);


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
