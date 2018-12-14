#ifdef __MPI__

#include <string>
#include "io/impl/mpi_module.h"

REGISTER_MODULE(MPIModule, "mpi");
REGISTER_MODULE(MPILockstepModule, "mpi lockstep");

/* Identifies keys where one value is a subset of the other */
static std::map<int, int> find_subsets(std::map<int, std::vector<int>> tags) {
    auto output = std::map<int, int>();

    // Check for overlap between destination data
    std::map<int, int> subsets;
    for (auto& pair : tags) {
        int dest = pair.first;
        auto& ltags = pair.second;

        // If this has not been assigned a subset
        if (subsets.find(dest) == subsets.end()) {
            for (auto& other : tags) {
                int other_dest = other.first;
                auto& other_ltags = other.second;

                // If this is a subset of other, assign this
                if (std::includes(
                        other_ltags.begin(), other_ltags.end(),
                        ltags.begin(), ltags.end())) {

                    // Break symmetry on equivalence using tag
                    if (other_ltags.size() != ltags.size()
                            or other_dest < dest)
                        subsets[dest] = other_dest;

                    break;
                }
            }
        }
    }

    // Jump pointers to find topmost parents
    std::map<int, int> parent_subsets;
    for (auto& pair : subsets) {
        int subset = pair.first;
        int superset = pair.second;

        while (true) {
            auto it = subsets.find(superset);
            if (it != subsets.end())
                superset = (*it).second;
            else break;
        }

        output[subset] = superset;
    }

    // Set the indices for the topmost parents to themselves
    for (auto& pair : tags)
        if (output.find(pair.first) == output.end())
            output[pair.first] = pair.first;

    return output;
}

MPIModule::MPIModule(LayerList layers, ModuleConfig *config)
        : Module(layers, config) {
    // Each layer should have a specified and unique IO type (input or output)
    enforce_specified_io_type("mpi");
    enforce_unique_io_type("mpi");

    this->mpi_rank = mpi_wrap_get_rank();
    int mpi_size = mpi_wrap_get_size();

    for (auto layer : layers) {
        auto layer_config = config->get_layer(layer);

        // Get tag
        int tag = layer_config->get_int("mpi tag", -1);
        if (tag < 0)
            LOG_ERROR("Unspecified MPI tag for layer "
                + layer->str() + " in MPIModule!");
        layer_tags[tag] = layer;

        // Get source/dest
        if (get_io_type(layer) == INPUT) {
            int source = layer_config->get_int("mpi source", -1);

            // Validate source
            if (source < 0 or source == mpi_rank or source >= mpi_size)
                LOG_ERROR("Unspecified/invalid MPI source for layer "
                    + layer->str() + " in MPIModule!");

            input_tags[source].push_back(tag);
        } else {
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

            // Validate destinations
            for (auto dest : dests) {
                if (dest < 0 or dest == mpi_rank or dest >= mpi_size)
                    LOG_ERROR("Unspecified/invalid MPI destination for layer "
                        + layer->str() + " in MPIModule!");
                output_tags[dest].push_back(tag);
            }
        }
    }

    // Create input buffers
    for (auto& pair : input_tags) {
        int source = pair.first;
        auto& ltags = pair.second;

        // Create buffer for the source
        int size = 0;
        for (auto& layer : ltags)
            size += layer_tags[layer]->size;
        input_buffers[source] = Pointer<float>(size, 0.0);

        // Sort layer tags (ensures consistency across ranks)
        std::sort(ltags.begin(), ltags.end());
    }

    // Sort output layer tags (ensures consistency across ranks)
    for (auto& pair : output_tags)
        std::sort(pair.second.begin(), pair.second.end());

    // Identify destinations whose layers are subsets of one another
    // This permits reuse of buffers
    this->buffer_indices = find_subsets(output_tags);

    // Create output buffers
    for (auto& pair : buffer_indices) {
        int sub_tag = pair.first;
        int super_tag = pair.second;

        auto& ltags = output_tags[sub_tag];

        // Compute buffer size
        int size = 0;
        for (auto& layer : ltags)
            size += layer_tags[layer]->size;
        buffer_sizes[sub_tag] = size;

        // If superset, create buffer
        // Otherwise, determine offset
        if (sub_tag == super_tag) {
            output_buffers[sub_tag] = Pointer<float>(size, 0.0);
            buffer_offsets[sub_tag] = 0;
        } else {
            int stop = output_tags[super_tag][0];
            int index = 0;
            int offset = 0;

            while (ltags[index] != stop)
                offset += layer_tags[ltags[index++]]->size;

            buffer_offsets[sub_tag] = offset;
        }
    }

    // Create requests
    recv_requests_array = mpi_wrap_create_request_array(input_tags.size());
    int index = 0;
    for (auto& pair : input_tags) {
        recv_requests[pair.first] = index;
        recv_requests_inv[index] = pair.first;
        index++;
    }

    for (auto& pair : output_tags)
        send_requests[pair.first] = mpi_wrap_create_request();


    // Report total sizes
    int total_input = 0;
    for (auto pair : input_tags)
        total_input += input_buffers[pair.first].get_size();

    int total_output = 0;
    for (auto pair : buffer_sizes)
        total_output += pair.second;

    int out_buf_size = 0;
    for (auto pair : output_buffers)
        out_buf_size += pair.second.get_size();

    LOG_DEBUG("MPI rank " + std::to_string(mpi_rank) +
        " send/recv: (" + std::to_string(total_input) +
        ", " + std::to_string(total_output) +
        ") out_buf_size = " + std::to_string(out_buf_size));
}

MPILockstepModule::MPILockstepModule(LayerList layers, ModuleConfig *config)
        : MPIModule(layers, config) {
    // Perform initial sends
    for (auto& pair : output_tags) {
        int dest = pair.first;
        auto req = send_requests[dest];

        LOG_DEBUG("MPI send from " + std::to_string(mpi_rank) +
            " to " + std::to_string(dest));

        auto& buf = output_buffers[buffer_indices[dest]];
        mpi_wrap_isend(req, buf + buffer_offsets[dest], buffer_sizes[dest], dest, 0);
    }
}

void MPIModule::feed_input_impl(Buffer *buffer) {
    // Initiate message receive
    for (auto& pair : input_tags) {
        int source = pair.first;
        auto& ltags = pair.second;
        auto& buf = input_buffers[source];

        LOG_DEBUG("MPI receive in " + std::to_string(mpi_rank) +
            " from " + std::to_string(source));
        mpi_wrap_irecv(recv_requests_array, recv_requests[source],
            buf, buf.get_size(), source, 0);
    }

    // Wait for messages
    int req_size = input_tags.size();
    for (int i = 0 ; i < req_size ; ++i) {
        int source = recv_requests_inv[mpi_wrap_wait_any(req_size, recv_requests_array)];
        auto& ltags = input_tags[source];
        auto& buf = input_buffers[source];

        // Distribute message to layer buffers
        int index = 0;
        for (auto& tag : ltags) {
            auto layer = layer_tags[tag];
            buf.slice(index, layer->size).copy_to(buffer->get_input(layer));
            index += layer->size;
        }
    }
}

void MPIModule::report_output_impl(Buffer *buffer) {
    // Wait for previous sends
    for (auto pair : send_requests) {
        LOG_DEBUG("MPI wait from " + std::to_string(mpi_rank) +
            " to " + std::to_string(pair.first));
        mpi_wrap_wait(pair.second);
    }

    // Copy data from layer buffers
    for (auto& pair : output_buffers) {
        auto& buf = pair.second;

        int index = 0;
        for (auto& layer_tag : output_tags[pair.first]) {
            auto layer = layer_tags[layer_tag];
            buffer->get_output(layer).cast<float>().copy_to(buf.slice(index, layer->size));
            index += layer->size;
        }
    }

    // Send messages
    for (auto& pair : output_tags) {
        int dest = pair.first;
        auto& buf = output_buffers[buffer_indices[dest]];
        auto req = send_requests[dest];

        LOG_DEBUG("MPI send from " + std::to_string(mpi_rank) +
            " to " + std::to_string(dest));
        mpi_wrap_isend(req, buf + buffer_offsets[dest], buffer_sizes[dest], dest, 0);
    }
}

void MPIModule::report(Report* report) { }

#endif
