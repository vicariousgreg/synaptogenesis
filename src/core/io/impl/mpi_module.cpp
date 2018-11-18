#ifdef __MPI__

#include <string>
#include "io/impl/mpi_module.h"

REGISTER_MODULE(MPIModule, "mpi");

MPIModule::MPIModule(LayerList layers, ModuleConfig *config)
        : Module(layers, config) {
    // Each layer should have a specified and unique IO type (input or output)
    enforce_specified_io_type("mpi");
    enforce_unique_io_type("mpi");

    int mpi_rank = mpi_wrap_get_rank();
    int mpi_size = mpi_wrap_get_size();

    for (auto layer : layers) {
        auto layer_config = config->get_layer(layer);

        // Retrieve lockstep flag
        // If true, an initial zero state will be sent
        // This corresponds to parallel structures/clusters (lock step)
        bool lockstep = layer_config->get_bool("lockstep", true);

        // Get tag
        int tag = layer_config->get_int("mpi tag", -1);
        if (tag < 0)
            LOG_ERROR("Unspecified MPI tag for layer "
                + layer->str() + " in MPIModule!");
        tags[layer] = tag;

        // Get source/dest
        if (get_io_type(layer) == INPUT) {
            int source = layer_config->get_int("mpi source", -1);
            if (source < 0 or source == mpi_rank or source >= mpi_size)
                LOG_ERROR("Unspecified/invalid MPI source for layer "
                    + layer->str() + " in MPIModule!");
            sources[layer] = source;
        } else {
            // Create local buffer, and destinations/requests arrays
            local_buffers[layer] = Pointer<float>(layer->size, 0.0);
            destinations[layer] = std::vector<int>();
            requests[layer] = std::vector<int>();

            // Extract destinations
            std::vector<int> dests;
            if (layer_config->has("mpi destinations"))
                dests.push_back(layer_config->get_int("mpi destinations", -1));
            else if (layer_config->has_array("mpi destinations"))
                for (auto d : layer_config->get_array("mpi destinations"))
                    dests.push_back(std::stoi(d));

            // Ensure nonzero set of destinations
            if (dests.size() == 0)
                LOG_ERROR("Unspecified MPI destinations for layer "
                    + layer->str() + " in MPIModule!");

            auto l_buf = local_buffers[layer];

            // Validate destinations, allocate resources, initial send
            for (auto dest : dests) {
                if (dest < 0 or dest == mpi_rank or dest >= mpi_size)
                    LOG_ERROR("Unspecified/invalid MPI destination for layer "
                        + layer->str() + " in MPIModule!");
                int req = mpi_wrap_create_request();
                requests[layer].push_back(req);
                destinations[layer].push_back(dest);

                LOG_DEBUG("Sending " + layer->str() + " to "
                    + std::to_string(dest) + " (" + std::to_string(tag) + ")");

                // Initiate first send if lockstep
                if (lockstep)
                    mpi_wrap_isend(req, l_buf, layer->size, dest, tag);
            }
        }
    }
}

void MPIModule::feed_input_impl(Buffer *buffer) {
    for (auto layer : layers) {
        if (get_io_type(layer) & INPUT) {
            int dest = sources[layer];
            int tag = tags[layer];

            LOG_DEBUG("Receiving " + layer->str() + " to "
                + std::to_string(dest) + " (" + std::to_string(tag) + ")");

            mpi_wrap_recv(buffer->get_input(layer).get(), layer->size, dest, tag);
        }
    }
}

void MPIModule::report_output_impl(Buffer *buffer) {
    for (auto layer : layers) {
        if (get_io_type(layer) & OUTPUT) {
            int tag = tags[layer];
            auto dests = destinations[layer];
            auto reqs = requests[layer];

            // Wait for previous sends
            for (auto req : reqs) {
                LOG_DEBUG("Waiting for  " + layer->str() +
                    " req " + std::to_string(req));
                mpi_wrap_wait(req);
            }

            // Copy buffer
            auto l_buf = local_buffers[layer];
            buffer->get_output(layer).cast<float>().copy_to(l_buf);

            // Send
            for (int i = 0 ; i < dests.size() ; ++i) {
                int dest = dests[i];
                int req = reqs[i];

                LOG_DEBUG("Sending " + layer->str() + " to "
                    + std::to_string(dest) + " (" + std::to_string(tag) + ")");

                mpi_wrap_isend(req, l_buf, layer->size, dest, tag);
            }
        }
    }
}

void MPIModule::report(Report* report) { }

#endif
