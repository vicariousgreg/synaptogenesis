#include <cstring>

#include "state/state.h"
#include "engine/kernel/kernel.h"
#include "util/tools.h"
#include "util/parallel.h"

State::State(Model *model)
        : model(model),
          attributes(build_attributes(model)) {
    /* Set up weight matrices */
    for (auto & conn : model->get_connections()) {
        WeightMatrix* matrix = new WeightMatrix(conn, attributes->get_matrix_depth(conn));
        this->weight_matrices[conn] = matrix;
        this->attributes->process_weight_matrix(matrix);
#ifdef PARALLEL
        matrix->send_to_device();
#endif
    }

    // Create the buffer
    int input_output_size = attributes->get_num_neurons(INPUT_OUTPUT);
    int input_size = input_output_size + attributes->get_num_neurons(INPUT);
    int output_size = input_output_size + attributes->get_num_neurons(OUTPUT);
    this->buffer = new Buffer(input_size, output_size, attributes->output_type); 

#ifdef PARALLEL
    cudaStreamCreate(&this->io_stream);
#endif
}

State::~State() {
    delete attributes;
    delete buffer;
    for (auto matrix : this->weight_matrices) delete matrix.second;

#ifdef PARALLEL
    cudaStreamDestroy(this->io_stream);
#endif
}
