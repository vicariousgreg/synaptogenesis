#include <climits>

#include "engine/engine.h"
#include "engine/context.h"
#include "engine/instruction.h"
#include "engine/cluster/cluster.h"
#include "network/network.h"
#include "io/buffer.h"
#include "io/environment.h"
#include "state/state.h"
#include "state/attributes.h"
#include "frontend.h"

Engine::Engine(Context *context, bool suppress_output)
        : context(context),
          learning_flag(true),
          suppress_output(suppress_output),
          refresh_rate(INT_MAX),
          time_limit(1.0 / refresh_rate),
          environment_rate(1),
          calc_rate(true) {
    rebuild();
}

void Engine::rebuild() {
    build_environment();
    build_clusters();
}

void Engine::build_environment() {
    /* Build environmental buffer */
    LayerList input_layers, expected_layers, output_layers;

    for (auto config : context->get_environment()->get_modules()) {
        // If output is suppressed, skip any pure output modules
        if (suppress_output and (Module::get_type(config) == OUTPUT)) continue;

        Module *module = Module::build_module(context->get_network(), config);
        modules.push_back(module);
        this->io_types[module->layer] |= module->get_type();
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

void Engine::build_clusters() {
    /* Build clusters */
    auto state = context->get_state();

    // Create the clusters and gather their nodes
    for (auto& structure : state->network->get_structures()) {
        auto cluster = build_cluster(
            structure, state, this);
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

Engine::~Engine() {
    delete buffer;
    for (auto module : modules) delete module;
    for (auto& cluster : clusters) delete cluster;
    for (auto& inst : inter_device_transfers) delete inst;
}

void Engine::stage_clear() {
    // Launch pre-input calculations
    for (auto& cluster : clusters)
        cluster->launch_pre_input_calculations();
}

void Engine::stage_input() {
    // Launch input transfer
    for (auto& cluster : clusters)
        cluster->launch_input();

    // Wait for input
    for (auto& cluster : clusters)
        cluster->wait_for_input();
}

void Engine::stage_calc() {
    for (auto& cluster : clusters) {
        // Launch post-input calculations
        cluster->launch_post_input_calculations();

        // Launch state update
        cluster->launch_state_update();

        // Launch weight updates
        if (learning_flag) cluster->launch_weight_update();
    }
}

void Engine::stage_output() {
    // Start output streaming
    for (auto& cluster : clusters)
        cluster->launch_output();

    // Wait for output
    for (auto& cluster : clusters)
        cluster->wait_for_output();
}

void Engine::step_input() {
    for (auto& module : this->modules)
        module->feed_input(buffer);

    for (auto& module : this->modules)
        module->feed_expected(buffer);
}

void Engine::step_output() {
    if (not suppress_output)
        for (auto& module : this->modules)
            module->report_output(buffer,
                Attributes::get_output_type(module->layer));
}

void Engine::ui_init() {
    Frontend::init_all();
}

void Engine::ui_launch() {
    Frontend::launch_all();
}

void Engine::ui_update() {
    Frontend::update_all(this->buffer);
}

void Engine::network_loop(int iterations, bool verbose) {
    run_timer.reset();

    for (int i = 0 ; i < iterations; ++i) {
        // Wait for timer, then start clearing inputs
        iteration_timer.reset();
        this->stage_clear();

        // Read sensory input
        sensory_lock.wait(NETWORK);
        this->stage_input();
        sensory_lock.pass(ENVIRONMENT);

        // Perform computations
        this->stage_calc();

        // Write motor output
        motor_lock.wait(NETWORK);
        // Use (i+1) because the locks belong to the Environment
        //   during the first iteration (Environment uses blank data
        //   on first iteration before the Engine sends any data)
        if ((i+1) % environment_rate == 0) this->stage_output();
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

void Engine::environment_loop(int iterations, bool verbose) {
    for (int i = 0; i < iterations; ++i) {
        // Write sensory buffer
        sensory_lock.wait(ENVIRONMENT);
        this->step_input();
        sensory_lock.pass(NETWORK);

        // Read motor buffer
        motor_lock.wait(ENVIRONMENT);
        if (i % environment_rate == 0) {
            // Stream output and update UI
            this->step_output();
            this->ui_update();
        }
        motor_lock.pass(NETWORK);
    }
}

Context* Engine::run(int iterations, bool verbose) {
    /**********************/
    /*** Initialization ***/
    /**********************/
    // Initialize cuda random states
    init_rand(context->get_network()->get_max_layer_size());

    // Set locks
    sensory_lock.set_owner(ENVIRONMENT);
    motor_lock.set_owner(NETWORK);

    // Start timer
    run_timer.reset();

    if (verbose) run_timer.query("Initialization");

    // Ensure device is synchronized without errors
    device_synchronize();
    device_check_error("Clock device synchronization failed!");
    if (verbose) device_check_memory();

    // Initialize UI
    this->ui_init();

    /**********************/
    /*** Launch threads ***/
    /**********************/
    std::thread network_thread(
        &Engine::network_loop, this, iterations, verbose);
    std::thread environment_thread(
        &Engine::environment_loop, this, iterations, verbose);

    // Launch UI on main thread
    this->ui_launch();

    // Wait for threads
    network_thread.join();
    environment_thread.join();

    /****************/
    /*** Clean up ***/
    /****************/
    free_rand();
    Frontend::cleanup();

    return context;
}
