#include <cstring>

#include "state/state.h"
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
            matrix->transfer_to_device();
        }
    }

#ifdef PARALLEL
    cudaStreamCreate(&this->io_stream);
#endif
}

State::~State() {
    for (auto att : attributes) delete att.second;
    for (auto matrix : this->weight_matrices) delete matrix.second;

#ifdef PARALLEL
    cudaStreamDestroy(this->io_stream);
#endif
}

std::string State::get_stream_cluster_name(Structure *structure) {
    return attributes.at(structure)->get_stream_cluster_name();
}

Pointer<float> State::get_input(Layer *layer, int register_index) const {
    return attributes.at(layer->structure)
        ->get_input(layer->id, register_index);
}

Pointer<Output> State::get_output(Layer *layer, int word_index) const {
    return attributes.at(layer->structure)
        ->get_output(layer->id, word_index);
}

int State::get_start_index(Layer *layer, int register_index) const {
    return attributes.at(layer->structure)
        ->get_start_index(layer->id, register_index);
}

const Attributes* State::get_attributes_pointer(Layer *layer) const {
    return attributes.at(layer->structure)->pointer;
}

const ATTRIBUTE_KERNEL State::get_attribute_kernel(Layer *layer) const {
    return attributes.at(layer->structure)->get_attribute_kernel();
}

Pointer<float> State::get_matrix(Connection* conn) const {
    return weight_matrices.at(conn)->get_data();
}

EXTRACTOR State::get_extractor(Connection *conn) const {
    return attributes.at(conn->to_layer->structure)->extractor;
}

SYNAPSE_KERNEL State::get_activator(Connection *conn) const {
    return attributes.at(conn->to_layer->structure)->get_activator(conn->type);
}

SYNAPSE_KERNEL State::get_updater(Connection *conn) const {
    return attributes.at(conn->to_layer->structure)->get_updater(conn->type);
}

OutputType State::get_output_type(Structure *structure) const {
    return attributes.at(structure)->output_type;
}
