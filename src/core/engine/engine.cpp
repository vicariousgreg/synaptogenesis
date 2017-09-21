#include <climits>

#include "engine/engine.h"
#include "engine/context.h"
#include "engine/report.h"
#include "engine/instruction.h"
#include "engine/cluster/cluster.h"
#include "network/network.h"
#include "io/buffer.h"
#include "io/environment.h"
#include "state/state.h"
#include "state/attributes.h"
#include "gui_controller.h"

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

void Engine::build_environment() {
    /* Build environmental buffer */
    LayerList input_layers, expected_layers, output_layers;

    for (auto config : context->get_environment()->get_modules()) {
        // Build module
        Module *module = Module::build_module(context->get_network(), config);
        modules.push_back(module);

        // Update io_types for all layers attached to the module
        for (auto layer : module->layers) {
            auto layer_io_type = io_types[layer];
            auto module_io_type = module->get_io_type(layer);

            if (module_io_type == 0)
                ErrorManager::get_instance()->log_error(
                    "Error in environment model:\n"
                    "  Error adding module to: " + layer->str() + "\n" +
                    "    Module has zero IO type for layer!");

            // Check for duplicate input/expected modules
            if (module_io_type & INPUT & layer_io_type)
                ErrorManager::get_instance()->log_error(
                    "Error in environment model:\n"
                    "  Error adding module to: " + layer->str() + "\n" +
                    "    Layer cannot have more than one input module!");
            if (module_io_type & EXPECTED & layer_io_type)
                ErrorManager::get_instance()->log_error(
                    "Error in environment model:\n"
                    "  Error adding module to: " + layer->str() + "\n" +
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

void Engine::rebuild() {
    clear();
    build_environment();
    build_clusters();
}

void Engine::clear() {
    delete buffer;

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

void Engine::network_loop(int iterations, bool verbose, Report** report) {
    run_timer.reset();

    for (int i = 0 ; i < iterations; ++i) {
        // Wait for timer, then start clearing inputs
        iteration_timer.reset();

        // Launch pre-input calculations
        for (auto& cluster : clusters)
            cluster->launch_pre_input_calculations();

        /**************************/
        /*** Read sensory input ***/
        /**************************/
        sensory_lock.wait(NETWORK);
        for (auto& cluster : clusters) cluster->launch_input();
        for (auto& cluster : clusters) cluster->wait_for_input();
        sensory_lock.pass(ENVIRONMENT);

        /****************************/
        /*** Perform computations ***/
        /****************************/
        for (auto& cluster : clusters) {
            cluster->launch_post_input_calculations();
            cluster->launch_state_update();
            if (learning_flag) cluster->launch_weight_update();
        }

        /**************************/
        /*** Write motor output ***/
        /**************************/
        motor_lock.wait(NETWORK);
        if (i % environment_rate == 0) {
            for (auto& cluster : clusters) cluster->launch_output();
            for (auto& cluster : clusters) cluster->wait_for_output();
        }
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

    // Create report
    *report = new Report(this, this->context->get_state(),
        iterations, run_timer.query(nullptr));

    // Report time if verbose
    if (verbose) {
        printf("Total time: %f\n", (*report)->total_time);
        printf("Time averaged over %d iterations: %f\n",
               iterations, (*report)->average_time);
    }
}

void Engine::environment_loop(int iterations, bool verbose) {
    for (int i = 0; i < iterations; ++i) {
        // Write sensory buffer
        sensory_lock.wait(ENVIRONMENT);
        for (auto& module : this->modules) {
            module->feed_input(buffer);
            module->feed_expected(buffer);
        }
        sensory_lock.pass(NETWORK);

        // Read motor buffer
        motor_lock.wait(ENVIRONMENT);
        if (i % environment_rate == 0) {
            // Stream output and update UI
            if (not suppress_output)
                for (auto& module : this->modules)
                    module->report_output(buffer);
            GuiController::get_instance()->update();
        }
        motor_lock.pass(NETWORK);

        // Cycle modules
        for (auto& module : this->modules) module->cycle();
    }
    // TODO: handle race conditions here
    GuiController::get_instance()->quit();
}

Context* Engine::run(int iterations, bool verbose) {
    // Initialize cuda random states
    init_rand(context->get_network()->get_max_layer_size());

    // Set locks
    sensory_lock.set_owner(ENVIRONMENT);
    motor_lock.set_owner(NETWORK);

    // Ensure device is synchronized without errors
    device_synchronize();
    device_check_error("Clock device synchronization failed!");
    if (verbose) device_check_memory();

    // Create
    Report *report = nullptr;

    // Launch threads
    std::thread network_thread(
        &Engine::network_loop, this, iterations, verbose, &report);
    std::thread environment_thread(
        &Engine::environment_loop, this, iterations, verbose);

    // Launch UI on main thread
    GuiController::get_instance()->launch();

    // Wait for threads
    network_thread.join();
    environment_thread.join();

    // Add report to context
    context->add_report(*report);
    delete report;

    // Clean up
    free_rand();

    return context;
}
