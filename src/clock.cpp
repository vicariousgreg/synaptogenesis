#include "clock.h"

void Clock::engine_loop() {
    for (int i = 0; i < iterations; ++i) {
        // Wait for clock signal, then start clearing inputs
        this->clock_lock.wait(DRIVER);
        this->engine->stage_clear();

        // Read sensory input
        this->sensory_lock.wait(DRIVER);
        this->engine->stage_input();
        this->sensory_lock.pass(ENVIRONMENT);

        // Calculate output
        this->engine->stage_calc_output();

        // Write motor output
        this->motor_lock.wait(DRIVER);
        this->engine->stage_send_output();
        this->motor_lock.pass(ENVIRONMENT);
        this->clock_lock.pass(CLOCK);

        // Finish computations
        this->engine->stage_remaining();
        this->engine->stage_weights();
    }
}

void Clock::environment_loop() {
    for (int i = 0; i < iterations; ++i) {
        // Write sensory buffer
        this->sensory_lock.wait(ENVIRONMENT);
        this->environment->step_input();
        this->sensory_lock.pass(DRIVER);

        // Compute

        this->motor_lock.wait(ENVIRONMENT);
        this->environment->step_output();
        if (i % this->environment_rate == 0) {
            // Read motor buffer
            this->environment->ui_update();
        }
        this->motor_lock.pass(DRIVER);
    }
}

void Clock::clock_loop() {
    // Run iterations
    this->run_timer.reset();
    this->iteration_timer.reset();
    for (int counter = 0; counter < iterations; ++counter) {
        this->iteration_timer.wait(this->time_limit);
        this->iteration_timer.reset();

        this->clock_lock.wait(CLOCK);
        this->clock_lock.pass(DRIVER);
    }
    this->clock_lock.wait(CLOCK);
    this->motor_lock.wait(ENVIRONMENT);
    this->sensory_lock.wait(ENVIRONMENT);

    // Report time if verbose
    if (verbose) {
        float time = this->run_timer.query("Total time");
        printf("Time averaged over %d iterations: %f\n", iterations, time/iterations);
    }

#ifdef PARALLEL
    check_memory();
#endif
}

void Clock::run(Model *model, int iterations, int environment_rate, bool verbose) {
    // Initialization
    this->sensory_lock.set_owner(ENVIRONMENT);
    this->motor_lock.set_owner(ENVIRONMENT);
    this->clock_lock.set_owner(CLOCK);

    // Build engine
    run_timer.reset();
    this->engine = new Engine(model);
    if (verbose) {
        printf("Built state.\n");
        run_timer.query("Initialization");
    }

    // Build environment and buffer
    this->environment = new Environment(model, this->engine->state->get_buffer());

    // Set iterations and verbose
    this->verbose = verbose;
    this->iterations = iterations;
    this->environment_rate = environment_rate;

#ifdef PARALLEL
    // Ensure device is synchronized without errors
    cudaCheckError("Clock device synchronization failed!");
#endif

    // Launch threads
    std::thread engine_thread(&Clock::engine_loop, this);
    std::thread environment_thread(&Clock::environment_loop, this);
    std::thread clock_thread(&Clock::clock_loop, this);

    engine_thread.join();
    environment_thread.join();
    clock_thread.join();

    delete this->engine;
    delete this->environment;
    this->engine = NULL;
    this->environment = NULL;
}
