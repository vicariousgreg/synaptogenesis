#include "state/impl/hebbian_rate_encoding_attributes.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/logger.h"

REGISTER_ATTRIBUTES(HebbianRateEncodingAttributes, "hebbian_rate_encoding", FLOAT)

/******************************************************************************/
/**************************** HEBBIAN LEARNING ********************************/
/******************************************************************************/

#define LEARNING_RATE 0.1

#define EXTRACT_DEST_OUT \
    float dest_out = extract(destination_outputs[to_index], 0);

#define EXTRACT_SOURCE_OUT \
    float source_out = extract(outputs[from_index], delay); \

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

CALC_CONVERGENT_CONVOLUTIONAL_BY_WEIGHT(update_hebbian_convergent_convolutional,
    ; ,

    INIT_WEIGHT_DELTA;,

    UPDATE_WEIGHT_DELTA;,

    UPDATE_WEIGHT_CONVOLUTIONAL;
);
CALC_DIVERGENT_CONVOLUTIONAL_BY_WEIGHT(update_hebbian_divergent_convolutional,
    ; ,

    INIT_WEIGHT_DELTA;,

    UPDATE_WEIGHT_DELTA;,

    UPDATE_WEIGHT_CONVOLUTIONAL;
);

Kernel<SYNAPSE_ARGS> HebbianRateEncodingAttributes::get_updater(
        Connection *conn) {
    if (conn->second_order)
        LOG_ERROR(
            "Second order plastic connections not supported!");

    if (conn->convolutional) {
        if (conn->get_type() == CONVERGENT)
            return get_update_hebbian_convergent_convolutional();
        else if (conn->get_type() == DIVERGENT)
            return get_update_hebbian_divergent_convolutional();
    }

    try {
        return update_hebbian_map.at(conn->get_type());
    } catch(std::out_of_range) { }

    // Log an error if the connection type is unimplemented
    LOG_ERROR("Unimplemented connection type!");
}

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

HebbianRateEncodingAttributes::HebbianRateEncodingAttributes(Layer *layer)
        : RateEncodingAttributes(layer) { }
