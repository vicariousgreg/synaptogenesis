#include <cstring>

#include "state/state.h"
#include "engine/kernel/kernel.h"
#include "util/tools.h"
#include "util/parallel.h"

State::State(Model *model)
        : model(model) {
    for (auto& structure : model->get_structures()) {
        auto att = build_attributes(structure);
        attributes[structure] = att;

        /* Set up weight matrices */
        for (auto & conn : structure->get_connections()) {
            WeightMatrix* matrix = new WeightMatrix(conn,
                att->get_matrix_depth(conn));
            this->weight_matrices[conn] = matrix;
            att->process_weight_matrix(matrix);
#ifdef PARALLEL
            matrix->send_to_device();
#endif
        }
        // Create the buffer
        this->buffers[structure] = new Buffer(structure, att->output_type);
    }

#ifdef PARALLEL
    cudaStreamCreate(&this->io_stream);
#endif
}

State::~State() {
    for (auto att : attributes) delete att.second;
    for (auto buffer : buffers) delete buffer.second;
    for (auto matrix : this->weight_matrices) delete matrix.second;

#ifdef PARALLEL
    cudaStreamDestroy(this->io_stream);
#endif
}

StreamCluster* State::build_stream_cluster(Structure *structure) {
    return attributes.at(structure)->build_stream_cluster(structure, this);
}

float* State::get_input(Layer *layer) const {
    return attributes.at(layer->structure)->get_input(layer->id);
}

OutputType State::get_output_type(Layer *layer) const {
    return attributes.at(layer->structure)->output_type;
}

Output* State::get_output(Layer *layer, int word_index) const {
    return attributes.at(layer->structure)->get_output(layer->id, word_index);
}

int State::get_start_index(Layer *layer) const {
    return attributes.at(layer->structure)->get_start_index(layer->id);
}

const Attributes* State::get_attributes_pointer(Layer *layer) const {
    return attributes.at(layer->structure)->pointer;
}

const ATTRIBUTE_KERNEL State::get_attribute_kernel(Layer *layer) const {
    return attributes.at(layer->structure)->get_attribute_kernel();
}

float*  State::get_matrix(Connection* conn) const {
    return weight_matrices.at(conn)->get_data();
}

EXTRACTOR State::get_extractor(Connection *conn) const {
    return attributes.at(conn->to_layer->structure)->extractor;
}

KERNEL State::get_activator(Connection *conn) const {
    return attributes.at(conn->to_layer->structure)->get_activator(conn->type);
}

KERNEL State::get_updater(Connection *conn) const {
    return attributes.at(conn->to_layer->structure)->get_updater(conn->type);
}

int State::get_num_neurons(Structure* structure) const {
    return attributes.at(structure)->total_neurons;
}

Buffer* State::get_buffer(Structure *structure) const {
    return buffers.at(structure);
}
