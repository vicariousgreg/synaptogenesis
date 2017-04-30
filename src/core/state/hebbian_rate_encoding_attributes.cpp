#include "state/hebbian_rate_encoding_attributes.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/error_manager.h"

REGISTER_ATTRIBUTES(HebbianRateEncodingAttributes, "hebbian_rate_encoding")

/******************************************************************************/
/**************************** HEBBIAN LEARNING ********************************/
/******************************************************************************/

#define LEARNING_RATE 0.1

#define EXTRACT_OUT \
    float dest_out = extractor(destination_outputs[to_index], 0);

#define UPDATE_WEIGHT \
    float source_out = extractor(outputs[from_index], delay); \
    float old_weight = weights[weight_index]; \
    weights[weight_index] = old_weight + \
        (LEARNING_RATE * source_out * \
            (dest_out - (source_out*old_weight))); \

#define UPDATE_WEIGHT_CONVOLUTIONAL \
    float weight_delta = 0.0; \
    float old_weight = weights[weight_index]; \
    for (int i = 0 ; i < to_size ; ++i) { \
        float source_out = extractor(outputs[from_index], delay); \
        weight_delta += source_out * \
            (dest_out - (source_out*old_weight)); \
    } \
    weights[weight_index] = old_weight + (LEARNING_RATE * weight_delta);

CALC_ALL(update_hebbian,
    ; ,
    EXTRACT_OUT;,
    UPDATE_WEIGHT;,
    ; );
CALC_ONE_TO_ONE(update_hebbian_convolutional,
    ; ,
    EXTRACT_OUT;, \
    UPDATE_WEIGHT_CONVOLUTIONAL;,
    ; );

Kernel<SYNAPSE_ARGS> HebbianRateEncodingAttributes::get_updater(
        Connection *conn, DendriticNode *node) {
    if (node->is_second_order())
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
