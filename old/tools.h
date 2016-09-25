#ifndef tools_h
#define tools_h

#include <ctime>

/* Calculates a random float between |fMin| and |fMax| */
float fRand(float fMin, float fMax);

/* Timer class.
 * Can be used to keep track of runtimes.
 * Call start() to set a start time.
 * Call stop() to calculate time difference since last start() call.
 */
class Timer {
    public:
        /* Sets a start time */
        void start();

        /* Calculates elapsed time since last start() call.
         * If |header| is provided, the time will be printed.
         */
        float stop(const char header[]);
        
    private:
        // Timestamp from last start call()
        clock_t start_time;
};

// Global variable for timer (singleton)
extern Timer timer;

#endif
