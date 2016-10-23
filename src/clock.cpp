#include <thread>
#include "clock.h"
#include "io/environment.h"
#include "driver/driver.h"

void driver_loop(Clock *clock, Driver *driver, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        // Wait for clock signal, then start clearing inputs
        clock->clock_lock.wait(DRIVER);
        driver->stage_clear();

        // Read sensory input
        clock->sensory_lock.wait(DRIVER);
        driver->stage_input();
        clock->sensory_lock.pass(ENVIRONMENT);

        // Calculate output
        driver->stage_calc_output();

        // Write motor output
        clock->motor_lock.wait(DRIVER);
        driver->stage_send_output();
        clock->motor_lock.pass(ENVIRONMENT);
        clock->clock_lock.pass(CLOCK);

        // Finish computations
        driver->stage_remaining();
    }
}

void environment_loop(Clock *clock, Environment *environment, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        // Write sensory buffer
        clock->sensory_lock.wait(ENVIRONMENT);
        environment->step_input();
        clock->sensory_lock.pass(DRIVER);

        // Compute

        // Write motor buffer
        clock->motor_lock.wait(ENVIRONMENT);
        environment->step_output();
        clock->motor_lock.pass(DRIVER);
    }
}

void Clock::run(Model *model, int iterations, bool verbose) {
    // Initialization
    this->sensory_lock.set_owner(ENVIRONMENT);
    this->motor_lock.set_owner(ENVIRONMENT);
    this->clock_lock.set_owner(CLOCK);

    // Build driver
    run_timer.reset();

    Driver *driver = build_driver(model);
    if (verbose) {
        printf("Built state.\n");
        run_timer.query("Initialization");
    }

    // Build environment and buffer
    Environment env(model, driver->state->get_buffer());

#ifdef PARALLEL
    // Ensure device is synchronized without errors
    cudaCheckError("Clock device synchronization failed!");
#endif

    // Launch threads
    std::thread driver_thread(driver_loop, this, driver, iterations);
    std::thread environment_thread(environment_loop, this, &env, iterations);

    // Run iterations
    run_timer.reset();
    iteration_timer.reset();
    for (int counter = 0; counter < iterations; ++counter) {
        iteration_timer.wait(this->time_limit);
        iteration_timer.reset();

        this->clock_lock.wait(CLOCK);
        this->clock_lock.pass(DRIVER);
    }

    driver_thread.join();
    environment_thread.join();

    // Report time if verbose
    if (verbose) {
        float time = run_timer.query("Total time");
        printf("Time averaged over %d iterations: %f\n", iterations, time/iterations);
    }

#ifdef PARALLEL
    check_memory();
#endif

    delete driver;
}
