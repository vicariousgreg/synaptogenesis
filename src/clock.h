#include <iostream>
#include <thread>
#include <mutex>
#include "model/model.h"
#include "io/buffer.h"
#include "tools.h"

enum Thread_ID {
    CLOCK,
    DRIVER,
    ENVIRONMENT
};

class Clock {
    public:
        Clock(float refresh_rate) : time_limit(1.0 / refresh_rate) {}
        void run(Model *model, int iterations, bool verbose);

        Buffer *buffer;
        Timer timer;
        float time_limit;

        std::mutex sensory_lock;
        Thread_ID sensory_owner;

        std::mutex motor_lock;
        Thread_ID motor_owner;

        std::mutex clock_lock;
        Thread_ID clock_owner;
};
