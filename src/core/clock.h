#include <iostream>
#include <thread>
#include <mutex>
#include <climits>

#include "model/model.h"
#include "io/buffer.h"
#include "util/tools.h"
#include "io/environment.h"
#include "engine/engine.h"

enum Thread_ID {
    CLOCK,
    DRIVER,
    ENVIRONMENT
};

class Lock {
    public:
        void set_owner(Thread_ID new_owner) {
            owner = new_owner;
        }

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
        Clock(float refresh_rate = INT_MAX, int environment_rate = 1, bool verbose = false)
                : time_limit(1.0 / refresh_rate),
                  environment_rate(environment_rate),
                  verbose(verbose) {}
        Clock() : time_limit(0), environment_rate(1), verbose(false) {}
        void run(Model *model, int iterations);

    private:
        void engine_loop(int iterations);
        void environment_loop(int iterations);

        int environment_rate;
        bool verbose;
        Engine *engine;
        Environment *environment;

        Timer run_timer;
        Timer iteration_timer;
        float time_limit;

        Lock sensory_lock;
        Lock motor_lock;
};
