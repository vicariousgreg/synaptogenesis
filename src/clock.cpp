#include "clock.h"

void Clock::driver_loop(Driver *driver, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        // Wait for clock signal, then start clearing inputs
        this->clock_lock.wait(DRIVER);
        driver->stage_clear();

        // Read sensory input
        this->sensory_lock.wait(DRIVER);
        driver->stage_input();
        this->sensory_lock.pass(ENVIRONMENT);

        // Calculate output
        driver->stage_calc_output();

        // Write motor output
        this->motor_lock.wait(DRIVER);
        driver->stage_send_output();
        this->motor_lock.pass(ENVIRONMENT);
        this->clock_lock.pass(CLOCK);

        // Finish computations
        driver->stage_remaining();
        driver->stage_weights();
    }
}

void Clock::environment_loop(Environment *environment, int iterations) {
    for (int i = 0; i < iterations; ++i) {
        // Write sensory buffer
        this->sensory_lock.wait(ENVIRONMENT);
        environment->step_input();
        this->sensory_lock.pass(DRIVER);

        // Compute

        // Write motor buffer
        this->motor_lock.wait(ENVIRONMENT);
        environment->step_output();
        this->motor_lock.pass(DRIVER);
    }
}

void Clock::clock_loop(int iterations, bool verbose) {
    // Run iterations
    this->run_timer.reset();
    this->iteration_timer.reset();
    for (int counter = 0; counter < iterations; ++counter) {
        this->iteration_timer.wait(this->time_limit);
        this->iteration_timer.reset();

        this->clock_lock.wait(CLOCK);
        this->clock_lock.pass(DRIVER);
    }

    // Report time if verbose
    if (verbose) {
        float time = this->run_timer.query("Total time");
        printf("Time averaged over %d iterations: %f\n", iterations, time/iterations);
    }

#ifdef PARALLEL
    check_memory();
#endif
}

void Clock::run(Model *model, int iterations, bool verbose) {
    // Initialization
    this->sensory_lock.set_owner(ENVIRONMENT);
    this->motor_lock.set_owner(ENVIRONMENT);
    this->clock_lock.set_owner(CLOCK);

    // Build driver
    run_timer.reset();

    Driver *driver = new Driver(model);
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
    std::thread driver_thread(&Clock::driver_loop, this, driver, iterations);
    std::thread environment_thread(&Clock::environment_loop, this, &env, iterations);
    std::thread clock_thread(&Clock::clock_loop, this, iterations, verbose);

    driver_thread.join();
    environment_thread.join();
    clock_thread.join();

    delete driver;
}
