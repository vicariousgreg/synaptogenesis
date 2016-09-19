#ifndef tools_h
#define tools_h

#include <ctime>

double fRand(double fMin, double fMax);

class Timer {
    public:
        void start();
        float stop(const char header[]);
        
    private:
        clock_t start_time;

};

extern Timer timer;

#endif
