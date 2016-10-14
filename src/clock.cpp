#include <thread>
#include "clock.h"

void wait_for_permission(std::mutex *mu, Thread_ID *owner, Thread_ID me) {
    while (true) {
        mu->lock();
        if (*owner == me) return;
        else mu->unlock();
    }
}

void driver_loop(Clock *clock, Driver *driver, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        /* Read sensory buffer */
        // Wait for sensory and clock locks
        wait_for_permission(&(clock->sensory_lock), &(clock->sensory_owner), DRIVER);
        wait_for_permission(&(clock->clock_lock), &(clock->clock_owner), DRIVER);

        // Read sensory input
        std::cout << "DRIVER READ\n";
        driver->step_input();

        // Pass sensory ownership back to environment
        clock->sensory_owner = ENVIRONMENT;
        clock->sensory_lock.unlock();

        /* Compute */
        driver->step_connections();

        /* Write motor buffer */
        // Wait for motor lock
        wait_for_permission(&(clock->motor_lock), &(clock->motor_owner), DRIVER);

        // Write motor output
        std::cout << "DRIVER WRITE\n";
        driver->step_output();
        driver->print_output();

        // Pass motor ownership back to environment
        clock->motor_owner = ENVIRONMENT;
        clock->motor_lock.unlock();

        // Pass clock ownership back to clock
        clock->clock_owner = CLOCK;
        clock->clock_lock.unlock();
    }
}

void environment_loop(Clock *clock, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        // Write sensory buffer
        wait_for_permission(&(clock->sensory_lock), &(clock->sensory_owner), ENVIRONMENT);
        std::cout << "ENVIRONMENT WRITE\n";

        clock->sensory_owner = DRIVER;
        clock->sensory_lock.unlock();

        // Compute

        // Write motor buffer
        wait_for_permission(&(clock->motor_lock), &(clock->motor_owner), ENVIRONMENT);
        std::cout << "ENVIRONMENT READ\n";

        clock->motor_owner = DRIVER;
        clock->motor_lock.unlock();
    }
}

void Clock::run(Driver *driver, int iterations) {
    this->sensory_owner = ENVIRONMENT;
    this->motor_owner = ENVIRONMENT;
    this->clock_owner = CLOCK;

    std::thread driver_thread(driver_loop, this, driver, iterations);
    std::thread environment_thread(environment_loop, this, iterations);

    this->timer.start();
    for (int counter = 0; counter < iterations; ++counter) {
        this->timer.wait(this->time_limit);
        this->timer.start();

        wait_for_permission(&(this->clock_lock), &(this->clock_owner), CLOCK);
        this->clock_owner = DRIVER;
        this->clock_lock.unlock();
    }

    driver_thread.join();
    environment_thread.join();
}
