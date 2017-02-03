#include "clock.h"

void Clock::engine_loop(int iterations, bool verbose) {
    this->run_timer.reset();

    for (int i = 0 ; i < iterations; ++i) {
        // Wait for timer, then start clearing inputs
        this->iteration_timer.reset();
        this->engine->stage_clear();

        // Write motor output
        this->motor_lock.wait(DRIVER);
        // Use (i+1) because the locks belong to the Environment
        //   during the first iteration (Environment uses blank data
        //   on first iteration before the Engine sends any data)
        if ((i+1) % this->environment_rate == 0) {
            //if (verbose) printf("Sending output... %d\n", i);
            this->engine->stage_output();
        }
        this->motor_lock.pass(ENVIRONMENT);

        // Read sensory input
        this->sensory_lock.wait(DRIVER);
        this->engine->stage_input();
        this->sensory_lock.pass(ENVIRONMENT);

        // Finish computations
        this->engine->stage_calc();

        // Synchronize with the clock
        this->iteration_timer.wait(this->time_limit);

        // Set the refresh rate if calc_rate is true
        if (this->calc_rate and i == 9) {
            this->time_limit = (run_timer.query(NULL)*1.1) / (i+1);
            this->refresh_rate = 1.0 / this->time_limit;
            if (verbose)
                printf("Updated refresh rate to %.2f fps\n", this->refresh_rate);
        }
    }

    // Report time if verbose
    if (verbose) {
        float time = this->run_timer.query("Total time");
        printf("Time averaged over %d iterations: %f\n",
            iterations, time/iterations);
    }
}

void Clock::environment_loop(int iterations, bool verbose) {
    for (int i = 0; i < iterations; ++i) {
        // Write sensory buffer
        this->sensory_lock.wait(ENVIRONMENT);
        this->environment->step_input();
        this->sensory_lock.pass(DRIVER);

        // Compute

        this->motor_lock.wait(ENVIRONMENT);
        if (i % this->environment_rate == 0) {
            // Stream output and update UI
            //if (verbose) printf("Updating UI... %d\n", i);
            this->environment->step_output();
            this->environment->ui_update();
        }
        this->motor_lock.pass(DRIVER);
    }
}

void Clock::run(Model *model, int iterations, bool verbose) {
    // Ensure model is built
    model->build();

    // Initialization
    this->sensory_lock.set_owner(ENVIRONMENT);
    this->motor_lock.set_owner(ENVIRONMENT);

    // Build engine
    run_timer.reset();
    this->engine = new Engine(model);
    //this->engine->set_learning_flag(false);  // disable learning
    if (verbose) {
        printf("Built state.\n");
        run_timer.query("Initialization");
    }

    // Build environment
    this->environment = new Environment(model, this->engine->get_buffer());

#ifdef PARALLEL
    // Ensure device is synchronized without errors
    cudaSync();
    cudaCheckError("Clock device synchronization failed!");
    check_memory();
#endif

    // Launch threads
    std::thread engine_thread(
        &Clock::engine_loop, this, iterations, verbose);
    std::thread environment_thread(
        &Clock::environment_loop, this, iterations, verbose);

    // Launch UI
    this->environment->ui_launch();

    // Wait for threads
    engine_thread.join();
    environment_thread.join();

    // Free memory for engine and environment
    delete this->engine;
    delete this->environment;
    this->engine = NULL;
    this->environment = NULL;
}
