#include <string>

#include "state/rate_encoding_attributes.h"
#include "util/tools.h"
#include "util/error_manager.h"
#include "util/parallel.h"

static RateEncodingParameters create_parameters(std::string str) {
    return RateEncodingParameters(0.0);

    //ErrorManager::get_instance()->log_error(
    //    "Unrecognized parameter string: " + str);
}

RateEncodingAttributes::RateEncodingAttributes(Model* model) : Attributes(model, FLOAT) {
    this->neuron_parameters = (RateEncodingParameters*)
        allocate_host(total_neurons, sizeof(RateEncodingParameters));

    // Fill in table
    for (auto& layer : model->get_layers()) {
        RateEncodingParameters params = create_parameters(layer->params);
        for (int j = 0 ; j < layer->size ; ++j)
            neuron_parameters[layer->get_start_index()+j] = params;
    }
}

RateEncodingAttributes::~RateEncodingAttributes() {
#ifdef PARALLEL
    cudaFree(this->neuron_parameters);
#else
    free(this->neuron_parameters);
#endif
}

void RateEncodingAttributes::process_weight_matrix(WeightMatrix* matrix) {
    Connection *conn = matrix->connection;
    float *mData = matrix->get_data();
    if (conn->plastic) {
        int num_weights = conn->get_num_weights();

        // Baseline
        transfer_weights(mData, mData + num_weights, num_weights);

        // Trace
        clear_weights(mData + 2*num_weights, num_weights);
    }
}


#ifdef PARALLEL
void RateEncodingAttributes::send_to_device() {
    Attributes::send_to_device();

    // Allocate space on GPU and copy data
    RateEncodingParameters* device_params = (RateEncodingParameters*)
        allocate_device(total_neurons, sizeof(RateEncodingParameters),
        this->neuron_parameters);
    free(this->neuron_parameters);
    this->neuron_parameters = device_params;
}
#endif
