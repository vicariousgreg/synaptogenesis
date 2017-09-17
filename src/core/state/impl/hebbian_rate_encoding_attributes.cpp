#include "state/impl/hebbian_rate_encoding_attributes.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/error_manager.h"

REGISTER_ATTRIBUTES(HebbianRateEncodingAttributes, "hebbian_rate_encoding", FLOAT)

/******************************************************************************/
/**************************** HEBBIAN LEARNING ********************************/
/******************************************************************************/

#define LEARNING_RATE 0.1

#define EXTRACT_DEST_OUT \
    float dest_out = extractor(destination_outputs[to_index], 0);

#define EXTRACT_SOURCE_OUT \
    float source_out = extractor(outputs[from_index], delay); \

#define UPDATE_WEIGHT \
    EXTRACT_SOURCE_OUT \
    float old_weight = weights[weight_index]; \
    weights[weight_index] = old_weight + \
        (LEARNING_RATE * source_out * \
            (dest_out - (source_out*old_weight))); \


CALC_ALL(update_hebbian,
    ; ,
    EXTRACT_DEST_OUT;,
    UPDATE_WEIGHT;,
    ; );

#define INIT_WEIGHT_DELTA \
    float weight_delta = 0.0; \
    float old_weight = weights[weight_index];

#define UPDATE_WEIGHT_DELTA \
    EXTRACT_DEST_OUT \
    EXTRACT_SOURCE_OUT \
    weight_delta += source_out * \
        (dest_out - (source_out*old_weight));

#define UPDATE_WEIGHT_CONVOLUTIONAL \
    weights[weight_index] = old_weight + (LEARNING_RATE * weight_delta);

CALC_CONVOLUTIONAL_BY_WEIGHT(update_hebbian_convolutional,
    ; ,

    INIT_WEIGHT_DELTA;,

    UPDATE_WEIGHT_DELTA;,

    UPDATE_WEIGHT_CONVOLUTIONAL;
);

Kernel<SYNAPSE_ARGS> HebbianRateEncodingAttributes::get_updater(
        Connection *conn) {
    if (conn->second_order)
        ErrorManager::get_instance()->log_error(
            "Second order plastic connections not supported!");

    switch (conn->type) {
        case FULLY_CONNECTED:
            return get_update_hebbian_fully_connected();
        case SUBSET:
            return get_update_hebbian_subset();
        case ONE_TO_ONE:
            return get_update_hebbian_one_to_one();
        case CONVERGENT:
            return get_update_hebbian_convergent();
        case CONVOLUTIONAL:
            return get_update_hebbian_convolutional();
        case DIVERGENT:
            return get_update_hebbian_divergent();
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
