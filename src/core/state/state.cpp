#include <cstring>

#include "state/state.h"
#include "util/tools.h"
#include "util/parallel.h"

State::State(Model *model)
        : model(model),
          buffer(new DeviceBuffer(model)) {
    for (auto neural_model : NeuralModels) {
        auto layers = model->get_layers(neural_model);
        if (layers.size() == 0) {
            attributes[neural_model] = nullptr;
        } else {
            auto att = build_attributes(layers, neural_model);
            attributes[neural_model] = att;

            /* Set up weight matrices */
            for (auto& layer : layers) {
                for (auto& conn : layer->get_input_connections()) {
                    WeightMatrix* matrix = new WeightMatrix(conn,
                        att->get_matrix_depth(conn));
                    this->weight_matrices[conn] = matrix;
                    att->process_weight_matrix(matrix);
                    matrix->transfer_to_device();
                }
            }
        }
    }
}

State::~State() {
    for (auto neural_model : NeuralModels)
        if (attributes[neural_model] != nullptr) delete attributes[neural_model];
    for (auto matrix : this->weight_matrices) delete matrix.second;
}

bool State::check_compatibility(Structure *structure) {
    // Retrieve represented neural models in the structure
    auto flags = structure->get_neural_model_flags();

    // Check relevant attributes for compatibility
    for (auto n : NeuralModels)
        if (flags[n] and
                not attributes[n]->check_compatibility(structure->cluster_type))
            return false;
    return true;
}

Pointer<float> State::get_input(Layer *layer, int register_index) const {
    return attributes[layer->neural_model]->get_input(layer->id, register_index);
}

Pointer<Output> State::get_output(Layer *layer, int word_index) const {
    return attributes[layer->neural_model]->get_output(layer->id, word_index);
}

int State::get_other_start_index(Layer *layer) const {
    return attributes[layer->neural_model]->get_other_start_index(layer->id);
}

const Attributes* State::get_attributes_pointer(Layer *layer) const {
    return attributes[layer->neural_model]->pointer;
}

Kernel<ATTRIBUTE_KERNEL> const State::get_attribute_kernel(Layer *layer) const {
    return attributes[layer->neural_model]->kernel;
}

Pointer<float> State::get_matrix(Connection* conn) const {
    return weight_matrices.at(conn)->get_data();
}

EXTRACTOR State::get_extractor(Connection *conn) const {
    return attributes[conn->from_layer->neural_model]->extractor;
}

Kernel<SYNAPSE_KERNEL>State::get_activator(Connection *conn) const {
    return attributes[conn->to_layer->neural_model]->get_activator(conn->type);
}

Kernel<SYNAPSE_KERNEL>State::get_updater(Connection *conn) const {
    return attributes[conn->to_layer->neural_model]->get_updater(conn->type);
}

OutputType State::get_output_type(Layer *layer) const {
    return attributes[layer->neural_model]->output_type;
}
