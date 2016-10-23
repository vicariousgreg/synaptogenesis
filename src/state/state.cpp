#include <cstdlib>
#include <cstring>
#include <sstream>

#include "state/state.h"
#include "tools.h"
#include "parallel.h"

State::State(Model *model, Attributes *attributes, int weight_depth)
        : attributes(attributes),
          weight_matrices(new WeightMatrices(model, weight_depth)) {
    int input_output_size = attributes->num_neurons[INPUT_OUTPUT];
    int input_size = input_output_size + attributes->num_neurons[INPUT];
    int output_size = input_output_size + attributes->num_neurons[OUTPUT];

    this->buffer = new Buffer(
        input_output_size + attributes->num_neurons[INPUT],
        input_output_size + attributes->num_neurons[OUTPUT],
        attributes->output_type);
}

State::~State() {
    delete this->weight_matrices;
    delete this->attributes;
    delete this->buffer;
}
