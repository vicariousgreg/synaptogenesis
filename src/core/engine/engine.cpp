#include <cfloat>

#include "engine/engine.h"
#include "engine/instruction.h"
#include "engine/cluster/cluster.h"
#include "network/network.h"
#include "io/buffer.h"
#include "io/environment.h"
#include "state/state.h"
#include "state/attributes.h"
#include "report.h"
#include "gui_controller.h"

Engine::Engine(Context context)
        : context(context),
          network_running(false),
          environment_running(false),
          learning_flag(true),
          suppress_output(false),
          refresh_rate(FLT_MAX),
          time_limit(0),
          environment_rate(1),
          iterations(0),
          verbose(false),
          buffer(nullptr),
          report(nullptr) { }

void Engine::build_environment(PropertyConfig args) {
    if (context.environment == nullptr) return;

    /* Build environmental buffer */
    LayerList input_layers, expected_layers, output_layers;

    for (auto config : context.environment->get_modules()) {
        // Skip if necessary
        if (config->get_bool("skip", false)) continue;

        // Build module
        Module *module = Module::build_module(
            context.network, new ModuleConfig(config));
        modules.push_back(module);

        // Update io_types for all layers attached to the module
        for (auto layer : module->layers) {
            auto layer_io_type = io_types[layer];
            auto module_io_type = module->get_io_type(layer);

            if (module_io_type == 0)
                LOG_ERROR(
                    "Error in environment model:\n"
                    "  Error adding module " + config->get("type") +
                    "to: " + layer->str() + "\n" +
                    "    Module has zero IO type for layer!");

            // Check for duplicate input/expected modules
            if (module_io_type & INPUT & layer_io_type)
                LOG_ERROR(
                    "Error in environment model:\n"
                    "  Error adding module " + config->get("type") +
                    "to: " + layer->str() + "\n" +
                    "    Layer cannot have more than one input module!");
            if (module_io_type & EXPECTED & layer_io_type)
                LOG_ERROR(
                    "Error in environment model:\n"
                    "  Error adding module " + config->get("type") +
                    "to: " + layer->str() + "\n" +
                    "    Layer cannot have more than one expected module!");

            this->io_types[layer] = layer_io_type | module_io_type;
        }
    }

    // Put layers in appropriate lists
    for (auto pair : io_types) {
        if (pair.second & INPUT)    input_layers.push_back(pair.first);
        if (pair.second & OUTPUT)   output_layers.push_back(pair.first);
        if (pair.second & EXPECTED) expected_layers.push_back(pair.first);
    }

    // Construct buffer
    buffer = build_buffer(
        ResourceManager::get_instance()->get_host_id(),
            input_layers, output_layers, expected_layers);
}

void Engine::build_clusters(PropertyConfig args) {
    /* Build clusters */
    auto state = context.state;

    // Create the clusters and gather their nodes
    for (auto& structure : state->network->get_structures()) {
        auto cluster = build_cluster(
            structure, state, this, args);
        clusters.push_back(cluster);
        for (auto& node : cluster->get_nodes())
            cluster_nodes[node->to_layer] = node;
    }

    // Add external dependencies to the nodes
    for (auto& cluster : clusters)
        cluster->add_external_dependencies(cluster_nodes);

    // Process inter-device instructions
    for (auto& cluster : clusters) {
        for (auto& node : cluster->get_nodes()) {
            for (auto& syn_inst : node->get_synapse_activate_instructions()) {
                auto conn = syn_inst->connection;

                // If inter-device, find or create corresponding transfer instruction
                if (state->is_inter_device(conn)) {
                    InterDeviceTransferInstruction *inst = nullptr;

                    // Search for existing instruction
                    for (auto inter_inst : this->inter_device_transfers)
                        if (inter_inst->matches(conn, state))
                            inst = inter_inst;

                    // Create if doesn't exist
                    // Clusters are responsible for handling these transfers,
                    //   since different types handle them differently
                    if (inst == nullptr) {
                        inst = new InterDeviceTransferInstruction(conn, state);
                        this->inter_device_transfers.push_back(inst);
                        cluster->add_inter_device_instruction(syn_inst, inst, true);
                    } else {
                        cluster->add_inter_device_instruction(syn_inst, inst, false);
                    }
                }
            }
        }
    }
}

void Engine::rebuild(PropertyConfig args) {
    clear();
    build_environment(args);
    build_clusters(args);
}

void Engine::clear() {
    if (buffer != nullptr) {
        delete buffer;
        buffer = nullptr;
    }

    // Clear modules and IO types
    for (auto module : modules) delete module;
    modules.clear();
    io_types.clear();

    // Clear clusters
    for (auto cluster : clusters) delete cluster;
    clusters.clear();
    for (auto& inst : inter_device_transfers) delete inst;
    inter_device_transfers.clear();

    // Clear resources
    ResourceManager::get_instance()->delete_streams();
    ResourceManager::get_instance()->delete_events();
}

Engine::~Engine() {
    clear();
}

size_t Engine::get_buffer_bytes() const {
    size_t size = 0;
    for (auto ptr : buffer->get_pointers())
        size += ptr->get_bytes();
    return size;
}

void Engine::single_thread_loop() {
    run_timer.reset();

    for (size_t i = 0 ; iterations == 0 or i < iterations; ++i) {
        // Wait for timer, then start clearing inputs
        if (time_limit > 0) iteration_timer.reset();

        // Launch pre-input calculations
        for (auto& cluster : clusters)
            cluster->launch_pre_input_calculations();

        /**************************/
        /*** Read sensory input ***/
        /**************************/
        for (auto& module : this->modules) {
            module->feed_input(buffer);
            module->feed_expected(buffer);
        }
        for (auto& cluster : clusters) cluster->launch_input(buffer);
        for (auto& cluster : clusters) cluster->wait_for_input();

        /****************************/
        /*** Perform computations ***/
        /****************************/
        for (auto& cluster : clusters)
            cluster->launch_post_input_calculations();
        for (auto& cluster : clusters)
            cluster->launch_state_update();
        if (learning_flag)
            for (auto& cluster : clusters)
                cluster->launch_weight_update();

        /**************************/
        /*** Write motor output ***/
        /**************************/
        if (i % environment_rate == 0) {
            for (auto& cluster : clusters) cluster->launch_output();
            for (auto& cluster : clusters) cluster->wait_for_output();
        }

        if (i % environment_rate == 0) {
            // Stream output and update UI
            if (not suppress_output)
                for (auto& module : this->modules)
                    module->report_output(buffer);
            GuiController::update();
        }

        // Cycle modules
        for (auto& module : this->modules) module->cycle();

        // Check for errors
        device_check_error(nullptr);

        // If engine gets interrupted, break
        if (not this->environment_running) {
            iterations = i;
            break;
        }

        // Print refresh rate if verbose
        if (verbose and i == 999)
            printf("Measured refresh rate: %.2f fps\n",
                1000 / (run_timer.query(nullptr)));

        // Synchronize with the clock
        if (time_limit > 0) iteration_timer.wait(time_limit);
    }

    // Set term lock owner to ensure interrupt doesn't hang
    term_lock.set_owner(NETWORK_THREAD);

    // Final synchronize
    device_synchronize();

    // Create report
    this->report = new Report(this, this->context.state,
        iterations, run_timer.query(nullptr));

    // Allow modules to modify report
    for (auto& module : this->modules)
        module->report(report);

    // Report report if verbose
    if (verbose) report->print();

    // Shutdown GUI
    GuiController::quit();
}

void Engine::network_loop() {
    run_timer.reset();

    for (size_t i = 0 ; iterations == 0 or i < iterations; ++i) {
        // Wait for timer, then start clearing inputs
        if (time_limit > 0) iteration_timer.reset();

        // Launch pre-input calculations
        for (auto& cluster : clusters)
            cluster->launch_pre_input_calculations();

        /**************************/
        /*** Read sensory input ***/
        /**************************/
        sensory_lock.wait(NETWORK_THREAD);
        for (auto& cluster : clusters) cluster->launch_input(buffer);
        for (auto& cluster : clusters) cluster->wait_for_input();
        sensory_lock.pass(ENVIRONMENT_THREAD);

        /****************************/
        /*** Perform computations ***/
        /****************************/
        for (auto& cluster : clusters)
            cluster->launch_post_input_calculations();
        for (auto& cluster : clusters)
            cluster->launch_state_update();
        if (learning_flag)
            for (auto& cluster : clusters)
                cluster->launch_weight_update();

        /**************************/
        /*** Write motor output ***/
        /**************************/
        motor_lock.wait(NETWORK_THREAD);
        if (i % environment_rate == 0) {
            for (auto& cluster : clusters) cluster->launch_output();
            for (auto& cluster : clusters) cluster->wait_for_output();
        }
        motor_lock.pass(ENVIRONMENT_THREAD);

        // Check for errors
        device_check_error(nullptr);

        // If engine gets interrupted, halt streams and break
        if (not this->network_running) {
            iterations = i;
            break;
        }

        // Print refresh rate if verbose
        if (verbose and i == 999)
            printf("Measured refresh rate: %.2f fps\n",
                1000 / (run_timer.query(nullptr)));


        // Synchronize with the clock
        if (time_limit > 0) iteration_timer.wait(time_limit);
    }

    // Wait for environment to terminate first
    term_lock.wait(NETWORK_THREAD);

    // Final synchronize
    device_synchronize();

    // Create report
    this->report = new Report(this, this->context.state,
        iterations, run_timer.query(nullptr));

    // Allow modules to modify report
    for (auto& module : this->modules)
        module->report(report);

    // Report report if verbose
    if (verbose) report->print();

    // Shutdown GUI
    GuiController::quit();
}

void Engine::environment_loop() {
    for (size_t i = 0 ; iterations == 0 or i < iterations; ++i) {
        // Write sensory buffer
        sensory_lock.wait(ENVIRONMENT_THREAD);
        for (auto& module : this->modules) {
            module->feed_input(buffer);
            module->feed_expected(buffer);
        }
        sensory_lock.pass(NETWORK_THREAD);

        // Read motor buffer
        motor_lock.wait(ENVIRONMENT_THREAD);
        if (i % environment_rate == 0) {
            // Stream output and update UI
            if (not suppress_output)
                for (auto& module : this->modules)
                    module->report_output(buffer);
            GuiController::update();
        }
        motor_lock.pass(NETWORK_THREAD);

        // Cycle modules
        for (auto& module : this->modules) module->cycle();

        // If engine gets interrupted, pass the locks and break
        if (not this->environment_running) {
            sensory_lock.pass(NETWORK_THREAD);
            motor_lock.pass(NETWORK_THREAD);
            this->network_running = false;
            break;
        }
    }

    // Pass the termination lock
    term_lock.pass(NETWORK_THREAD);
}

Report* Engine::run(PropertyConfig args) {
    // Set engine to active
    Engine::activate(this);

    // Transfer state to device
    // This renders the pointers in the engine outdated,
    //   so the engine must be rebuilt
    context.state->transfer_to_device();

    // Launch threads
    Scheduler::get_instance()->start(4);

    // Rebuild engine
    rebuild(args);

    // Initialize cuda random states
    init_rand(context.network->get_max_layer_size());

    // Extract parameters
    this->verbose = args.get_bool("verbose", false);
    this->learning_flag = args.get_bool("learning flag", true);
    this->suppress_output = args.get_bool("suppress output", false);
    this->environment_rate = args.get_int("environment rate", 1);
    this->refresh_rate = args.get_float("refresh rate", FLT_MAX);
    this->time_limit = (refresh_rate == FLT_MAX)
        ? 0 : (1.0 / this->refresh_rate);

    // If iterations is explicitly provided, use it
    if (args.has("iterations"))
        this->iterations = args.get_int("iterations", 1);
    else {
        // Otherwise, use the max of the expected iterations
        this->iterations = 0;
        for (auto module : modules)
            this->iterations =
                std::max(this->iterations, module->get_expected_iterations());
        if (this->iterations == 0)
            LOG_WARNING(
                "Unspecified number of iterations -- running indefinitely.");
    }

    network_running = true;
    environment_running = true;

    // Set locks
    sensory_lock.set_owner(ENVIRONMENT_THREAD);
    motor_lock.set_owner(NETWORK_THREAD);
    term_lock.set_owner(ENVIRONMENT_THREAD);

    // Ensure device is synchronized without errors
    device_synchronize();
    device_check_error("Clock device synchronization failed!");
    if (verbose) device_check_memory();

    std::vector<std::thread> threads;
    if (args.get_bool("multithreaded", true)) {
        if (verbose) printf("\nLaunching multithreaded...\n\n");
        threads.push_back(std::thread(
            &Engine::network_loop, this));
        threads.push_back(std::thread(
            &Engine::environment_loop, this));
    } else {
        if (verbose) printf("\nLaunching single threaded...\n\n");
        threads.push_back(std::thread(
            &Engine::single_thread_loop, this));
    }

    // Launch UI on main thread
    GuiController::launch();

    // Wait for threads
    for (auto& thread : threads)
        thread.join();

    // Shutdown the Scheduler
    Scheduler::get_instance()->shutdown();

    // Set engine to inactive
    Engine::deactivate(this);

    // Clean up
    free_rand();
    network_running = false;
    environment_running = false;

    auto report = this->report;
    this->report = nullptr;

    report->set_child("args", &args);
    return report;
}

std::set<Engine*> Engine::active_engines;

void Engine::activate(Engine* engine) {
    active_engines.insert(engine);
}

void Engine::deactivate(Engine* engine) {
    if (active_engines.count(engine))
        active_engines.erase(engine);
}

void Engine::interrupt() {
    for (auto engine : active_engines) {
        if (engine->verbose) printf("Interrupting engine...\n");

        // Stop the environment
        engine->environment_running = false;

        // Wait for term lock
        while (engine->term_lock.get_owner() != NETWORK_THREAD);

        // Ensure network thread gets locks
        engine->sensory_lock.set_owner(NETWORK_THREAD);
        engine->motor_lock.set_owner(NETWORK_THREAD);
    }
    active_engines.clear();
}
