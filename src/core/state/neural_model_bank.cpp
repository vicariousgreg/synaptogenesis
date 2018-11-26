#include "state/neural_model_bank.h"
#include "state/attributes.h"
#include "state/weight_matrix.h"

NeuralModelBank *NeuralModelBank::instance = nullptr;

NeuralModelBank *NeuralModelBank::get_instance() {
    if (NeuralModelBank::instance == nullptr)
        NeuralModelBank::instance = new NeuralModelBank();
    return NeuralModelBank::instance;
}

const std::set<std::string> NeuralModelBank::get_neural_models() {
    return get_instance()->neural_models;
}

OutputType NeuralModelBank::get_output_type(std::string neural_model) {
    return get_instance()->output_types[neural_model];
}

bool NeuralModelBank::register_attributes(std::string neural_model,
        OutputType output_type, ATT_BUILD_PTR build_ptr) {
    auto bank = get_instance();

    if (bank->att_build_pointers.count(neural_model) == 1)
        LOG_ERROR(
            "Duplicate attributes neural model string: "
                + neural_model + "!");

    bank->neural_models.insert(neural_model);
    bank->att_build_pointers[neural_model] = build_ptr;
    bank->output_types[neural_model] = output_type;
    return true;
}

bool NeuralModelBank::register_weight_matrix(std::string neural_model,
        MAT_BUILD_PTR build_ptr) {
    auto bank = get_instance();

    if (bank->mat_build_pointers.count(neural_model) == 1)
        LOG_ERROR(
            "Duplicate weight matrix neural model string: "
                + neural_model + "!");

    bank->neural_models.insert(neural_model);
    bank->mat_build_pointers[neural_model] = build_ptr;
    return true;
}

Attributes* NeuralModelBank::build_attributes(Layer *layer) {
    auto neural_model = layer->neural_model;

    try {
        // If the layer is a ghost layer, use the corresponding ghost attributes
        if (layer->is_ghost) {
            auto output_type = get_output_type(neural_model);

            switch (output_type) {
                case FLOAT: neural_model = "ghost float"; break;
                case BIT:   neural_model = "ghost bit";   break;
                case INT:   neural_model = "ghost int";   break;
            }
        }

        return get_instance()
            ->att_build_pointers.at(neural_model)(layer);
    } catch (std::out_of_range) {
        LOG_ERROR("Unrecognized neural model string: " + neural_model + "!");
    }
}

WeightMatrix* NeuralModelBank::build_weight_matrix(Connection *conn) {
    auto neural_model = conn->to_layer->neural_model;
    try {
        return get_instance()
            ->mat_build_pointers.at(neural_model)(conn);
    } catch (std::out_of_range) {
        LOG_DEBUG("Using default weight matrix for: " + neural_model + "!");

        // Use default weight matrix if none specified
        return WeightMatrix::build(conn);
    }
}
