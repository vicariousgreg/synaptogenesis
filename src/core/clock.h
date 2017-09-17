#ifndef clock_h
#define clock_h

#include <iostream>
#include <thread>
#include <mutex>
#include <climits>
#include <string>

#include "util/tools.h"
#include "context.h"

class Model;
class State;
class Environment;
class Engine;

enum Thread_ID {
    ENGINE,
    ENVIRONMENT
};

class Lock {
    public:
        void set_owner(Thread_ID new_owner)
            { owner = new_owner; }

        void wait(Thread_ID me) {
            while (true) {
                mutex.lock();
                if (owner == me) return;
                else mutex.unlock();
                std::this_thread::yield();
            }
        }

        void pass(Thread_ID new_owner) {
            owner = new_owner;
            mutex.unlock();
        }

    private:
        std::mutex mutex;
        Thread_ID owner;
};

class Clock {
    public:
        Clock(bool calc_rate, int environment_rate = 1)
                : refresh_rate(INT_MAX),
                  time_limit(1.0 / refresh_rate),
                  environment_rate(environment_rate),
                  calc_rate(calc_rate),
                  suppress_output(false),
                  learning_flag(true) { }

        Clock(float refresh_rate, int environment_rate = 1)
                : refresh_rate(refresh_rate),
                  time_limit(1.0 / refresh_rate),
                  environment_rate(environment_rate),
                  calc_rate(false),
                  suppress_output(false),
                  learning_flag(true) { }

        bool set_suppress_output(bool status) { suppress_output = status; }
        bool set_learning_flag(bool status) { learning_flag = status; }

        Context* run(Context *context, int iterations, bool verbose);

    private:
        void engine_loop(int iterations, bool verbose);
        void environment_loop(int iterations, bool verbose);

        int environment_rate;
        Engine *engine;
        Environment *environment;

        Timer run_timer;
        Timer iteration_timer;
        float refresh_rate, time_limit;
        bool calc_rate;
        bool suppress_output;
        bool learning_flag;

        Lock sensory_lock;
        Lock motor_lock;
};

#endif
