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

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

#include <math.h>

GLOBAL void re_attribute_kernel(const Attributes *att, int start_index, int count) {
    float *outputs = (float*)att->output;
    float *inputs = (float*)att->input;
    int total_neurons = att->total_neurons;

#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count) {
        nid += start_index;
#else
    for (int nid = start_index ; nid < start_index+count; ++nid) {
#endif
        float next_value = outputs[nid];
        int index;
        for (index = 0 ; index < HISTORY_SIZE-1 ; ++index) {
            float curr_value = next_value;
            next_value = outputs[total_neurons * (index + 1) + nid];
            outputs[total_neurons * index + nid] = next_value;
        }
        float input = inputs[nid];
        outputs[total_neurons * index + nid] =
            (input > 0.0) ? tanh(0.1*input) : 0.0;
    }
}
