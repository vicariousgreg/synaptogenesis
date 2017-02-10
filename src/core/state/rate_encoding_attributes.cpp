#include <string>

#include "state/rate_encoding_attributes.h"
#include "engine/engine.h"
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

Engine *RateEncodingAttributes::build_engine(Model *model, State *state) {
    return new FeedforwardEngine(model, state);
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
            //(input > 0.0) ? input : 0.0;
            //tanh(input);
    }
}

/******************************************************************************/
/**************************** HEBBIAN LEARNING ********************************/
/******************************************************************************/

#define LEARNING_RATE 0.1

#define EXTRACT_OUT(to_index) \
    float dest_out = extractor(destination_outputs[to_index], 0);

#define UPDATE_WEIGHT(from_index, weight_index, dest_out) \
    float source_out = extractor(outputs[from_index], delay); \
    float old_weight = weights[weight_index]; \
    weights[weight_index] = old_weight + \
        (LEARNING_RATE * source_out * \
            (dest_out - (source_out*old_weight))); \

#define UPDATE_WEIGHT_CONVOLUTIONAL(index) \
    EXTRACT_OUT(index); \
    float weight_delta = 0.0; \
    float old_weight = weights[index]; \
    for (int i = 0 ; i < to_size ; ++i) { \
        float source_out = extractor(outputs[index], delay); \
        weight_delta += source_out * \
            (dest_out - (source_out*old_weight)); \
    } \
    weights[index] = old_weight + (LEARNING_RATE * weight_delta);

CALC_FULLY_CONNECTED(update_fully_connected_hebbian,
    ; ,
    EXTRACT_OUT(to_index);,
    UPDATE_WEIGHT(from_index, weight_index, dest_out);,
    ; );
CALC_ONE_TO_ONE(update_one_to_one_hebbian,
    ; ,
    EXTRACT_OUT(index);
    UPDATE_WEIGHT(index, index, dest_out);
    );
CALC_CONVERGENT(update_convergent_hebbian,
    ; ,
    EXTRACT_OUT(to_index);,
    UPDATE_WEIGHT(from_index, weight_index, dest_out);,
    ; );
CALC_ONE_TO_ONE(update_convolutional_hebbian,
    ; ,
    UPDATE_WEIGHT_CONVOLUTIONAL(index);
    );

KERNEL RateEncodingAttributes::get_updater(ConnectionType conn_type) {
    switch (conn_type) {
        case FULLY_CONNECTED:
            return update_fully_connected_hebbian;
        case ONE_TO_ONE:
            return update_one_to_one_hebbian;
        case CONVERGENT:
            return update_convergent_hebbian;
        case CONVOLUTIONAL:
            return update_convolutional_hebbian;
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}
