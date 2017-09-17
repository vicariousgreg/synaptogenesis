#include "clock.h"
#include "model/model.h"
#include "io/environment.h"
#include "engine/engine.h"
#include "model/model.h"
#include "state/state.h"
#include "util/parallel.h"
#include "frontend.h"

void Clock::engine_loop(int iterations, bool verbose) {
    run_timer.reset();

    for (int i = 0 ; i < iterations; ++i) {
        // Wait for timer, then start clearing inputs
        iteration_timer.reset();
        engine->stage_clear();

        // Read sensory input
        sensory_lock.wait(ENGINE);
        engine->stage_input();
        sensory_lock.pass(ENVIRONMENT);

        // Perform computations
        engine->stage_calc();

        // Write motor output
        motor_lock.wait(ENGINE);
        // Use (i+1) because the locks belong to the Environment
        //   during the first iteration (Environment uses blank data
        //   on first iteration before the Engine sends any data)
        if ((i+1) % environment_rate == 0) engine->stage_output();
        motor_lock.pass(ENVIRONMENT);

        // Synchronize with the clock
        iteration_timer.wait(time_limit);

        // Set the refresh rate if calc_rate is true
        if (calc_rate and i == 999) {
            time_limit = (run_timer.query(nullptr)*1.1) / (i+1);
            refresh_rate = 1.0 / time_limit;
            if (verbose)
                printf("Updated refresh rate to %.2f fps\n", refresh_rate);
        }

        // Check for errors
        device_check_error(nullptr);
    }

    // Final synchronize
    device_synchronize();

    // Report time if verbose
    if (verbose) {
        float time = run_timer.query("Total time");
        printf("Time averaged over %d iterations: %f\n",
               iterations, time/iterations);
    }
}

void Clock::environment_loop(int iterations, bool verbose) {
    for (int i = 0; i < iterations; ++i) {
        // Write sensory buffer
        sensory_lock.wait(ENVIRONMENT);
        environment->step_input();
        sensory_lock.pass(ENGINE);

        // Read motor buffer
        motor_lock.wait(ENVIRONMENT);
        if (i % environment_rate == 0) {
            // Stream output and update UI
            environment->step_output();
            environment->ui_update();
        }
        motor_lock.pass(ENGINE);
    }
}

Context* Clock::run(Context *context, int iterations, bool verbose) {
    this->environment = context->environment;
    this->engine = context->engine;
    /**********************/
    /*** Initialization ***/
    /**********************/
    // Initialize cuda random states
    init_rand(context->model->get_max_layer_size());

    // Set locks
    sensory_lock.set_owner(ENVIRONMENT);
    motor_lock.set_owner(ENGINE);

    // Start timer
    run_timer.reset();

    if (verbose) run_timer.query("Initialization");

    // Ensure device is synchronized without errors
    device_synchronize();
    device_check_error("Clock device synchronization failed!");
    if (verbose) device_check_memory();

    // Initialize UI
    environment->ui_init();

    /**********************/
    /*** Launch threads ***/
    /**********************/
    std::thread engine_thread(
        &Clock::engine_loop, this, iterations, verbose);
    std::thread environment_thread(
        &Clock::environment_loop, this, iterations, verbose);

    // Launch UI on main thread
    environment->ui_launch();

    // Wait for threads
    engine_thread.join();
    environment_thread.join();

    /****************/
    /*** Clean up ***/
    /****************/
    engine = nullptr;
    environment = nullptr;

    free_rand();
    Frontend::cleanup();

    return context;
}
