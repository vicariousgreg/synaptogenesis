#include "state/hebbian_rate_encoding_attributes.h"
#include "util/error_manager.h"

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
CALC_DIVERGENT(update_divergent_hebbian,
    ; ,
    EXTRACT_OUT(to_index);,
    UPDATE_WEIGHT(from_index, weight_index, dest_out);,
    ; );

Kernel<SYNAPSE_ARGS> HebbianRateEncodingAttributes::get_updater(ConnectionType conn_type) {
    switch (conn_type) {
        case FULLY_CONNECTED:
            return get_update_fully_connected_hebbian();
        case ONE_TO_ONE:
            return get_update_one_to_one_hebbian();
        case CONVERGENT:
            return get_update_convergent_hebbian();
        case CONVOLUTIONAL:
            return get_update_convolutional_hebbian();
        case DIVERGENT:
            return get_update_divergent_hebbian();
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

HebbianRateEncodingAttributes::HebbianRateEncodingAttributes(LayerList &layers)
        : RateEncodingAttributes(layers) { }

HebbianRateEncodingAttributes::~HebbianRateEncodingAttributes() { }

void HebbianRateEncodingAttributes::schedule_transfer() {
    RateEncodingAttributes::schedule_transfer();
}

