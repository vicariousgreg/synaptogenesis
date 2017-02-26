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

RateEncodingAttributes::RateEncodingAttributes(Structure* structure)
        : Attributes(structure, FLOAT, re_attribute_kernel) {
    this->neuron_parameters = Pointer<RateEncodingParameters>(total_neurons);

    // Fill in table
    int start_index = 0;
    for (auto& layer : structure->get_layers()) {
        RateEncodingParameters params = create_parameters(layer->params);
        for (int j = 0 ; j < layer->size ; ++j)
            neuron_parameters[start_index+j] = params;
        start_index += layer->size;
    }
}

RateEncodingAttributes::~RateEncodingAttributes() {
    this->neuron_parameters.free();
}

void RateEncodingAttributes::transfer_to_device() {
    Attributes::transfer_to_device();

    this->neuron_parameters.transfer_to_device();
}

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

#include <math.h>

GLOBAL void re_attribute_kernel(const AttributeData attribute_data) {
    PREAMBLE_ATTRIBUTES;
    float *f_outputs = (float*)outputs;

#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < size) {
#else
    for (int nid = 0 ; nid < size; ++nid) {
#endif
        float next_value = f_outputs[nid];
        int index;
        for (index = 0 ; index < history_size-1 ; ++index) {
            float curr_value = next_value;
            next_value = f_outputs[size * (index + 1) + nid];
            f_outputs[size * index + nid] = next_value;
        }
        float input = inputs[nid];
        f_outputs[size * index + nid] =
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

SYNAPSE_KERNEL RateEncodingAttributes::get_updater(ConnectionType conn_type) {
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
