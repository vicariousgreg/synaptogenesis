#include <string>
#include <math.h>

#include "state/impl/leaky_izhikevich_attributes.h"
#include "state/weight_matrix.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/tools.h"

REGISTER_ATTRIBUTES(LeakyIzhikevichAttributes, "leaky_izhikevich", BIT)
bool __mat_dummy = NeuralModelBank::register_weight_matrix("leaky_izhikevich", IzhikevichWeightMatrix::build);

/******************************************************************************/
/************************** TRACE UPDATER KERNELS *****************************/
/******************************************************************************/

/* STDP A constant */
#define STDP_A_POS 0.4

#define STDP_TAU_POS       0.44   // tau = 1.8ms
#define STDP_TAU_NEG       0.833   // tau = 6ms

#define UPDATE_EXTRACTIONS \
    IzhikevichWeightMatrix *matrix = \
        (IzhikevichWeightMatrix*)synapse_data.matrix; \
    float *presyn_traces   = matrix->presyn_traces.get(); \
    float *eligibilities   = matrix->eligibilities.get(); \
    int   *delays = matrix->delays.get(); \
    int   *from_time_since_spike = matrix->time_since_spike.get(); \
\
    LeakyIzhikevichAttributes *att = \
        (LeakyIzhikevichAttributes*)synapse_data.attributes; \
    float *to_traces = att->postsyn_exc_trace.get(); \
    int   *to_time_since_spike = att->time_since_spike.get(); \
    float learning_rate = matrix->learning_rate; \

#define GET_DEST_ACTIVITY \
    float dest_exc_trace = to_traces[to_index]; \
    int   dest_time_since_spike = to_time_since_spike[to_index]; \
    float dest_spike = extract(destination_outputs[to_index], 0);

/* Minimum weight */
#define MIN_WEIGHT 0.0001

/* Time dynamics for long term eligibility trace */
#define C_A   0.000001
#define C_TAU 0.0001

#define UPDATE_WEIGHT \
    float weight = weights[weight_index]; \
\
    if (weight >= MIN_WEIGHT) { \
        /* Extract postsynaptic trace */ \
        float src_spike = extract(outputs[from_index], delays[weight_index]); \
        int src_time_since_spike = from_time_since_spike[from_index] = \
            ((src_spike > 0.0) \
                ? 0 \
                : MIN(32, from_time_since_spike[from_index] + 1)); \
\
        /* Update presynaptic trace */ \
        float src_trace = (opcode == ADD) \
            ? (presyn_traces[weight_index] = \
                (src_spike > 0.0) \
                    ? (presyn_traces[weight_index] + STDP_A_POS) \
                    : presyn_traces[weight_index] * STDP_TAU_POS) \
            : ((dest_spike > 0.0 and src_time_since_spike > 0 and src_time_since_spike < 32) \
                /* positive iSTDP function of delta T */ \
                ? powf(src_time_since_spike, 10) \
                  * 0.000001 \
                  * 2.29 \
                  * expf(-1.1 * src_time_since_spike) \
                : 0.0); \
\
        float dest_trace = (opcode == ADD) \
            ? dest_exc_trace \
            : ((src_spike > 0.0 and dest_time_since_spike > 0 and dest_time_since_spike < 32) \
                /* negative iSTDP function of delta T */ \
                ? powf(dest_time_since_spike, 10) \
                  * 0.0000001 \
                  * 2.6 \
                  * expf(-0.94 * dest_time_since_spike) \
                : 0.0); \
\
        /* Compute delta from short term dynamics */ \
        float c_delta = \
                (dest_spike * src_trace) \
                - (src_spike  * dest_trace); \
\
        float c = eligibilities[weight_index] + (learning_rate * c_delta); \
\
        /* Update eligibility trace */ \
        eligibilities[weight_index] -= (c - C_A) * C_TAU; \
\
        /* Calculate new weight */ \
        weight += c; \
\
        /* Ensure weight stays within boundaries */ \
        weights[weight_index] = weight = \
            MAX(MIN_WEIGHT, MIN(max_weight, weight)); \
    }

CALC_ALL(update_liz,
    UPDATE_EXTRACTIONS,
    GET_DEST_ACTIVITY,
    UPDATE_WEIGHT,
; );

Kernel<SYNAPSE_ARGS> LeakyIzhikevichAttributes::get_updater(Connection *conn) {
    // Second order and convolutional updaters are not currently supported
    if (conn->second_order or conn->convolutional)
        LOG_ERROR(
            "Unimplemented connection type!");

    std::map<ConnectionType, std::map<Opcode, Kernel<SYNAPSE_ARGS>>> funcs;
    funcs[FULLY_CONNECTED][ADD]  = get_update_liz_fully_connected();
    funcs[SUBSET][ADD]           = get_update_liz_subset();
    funcs[ONE_TO_ONE][ADD]       = get_update_liz_one_to_one();
    funcs[CONVERGENT][ADD]       = get_update_liz_convergent();
    funcs[DIVERGENT][ADD]        = get_update_liz_divergent();
    funcs[FULLY_CONNECTED][SUB]  = get_update_liz_fully_connected();
    funcs[SUBSET][SUB]           = get_update_liz_subset();
    funcs[ONE_TO_ONE][SUB]       = get_update_liz_one_to_one();
    funcs[CONVERGENT][SUB]       = get_update_liz_convergent();
    funcs[DIVERGENT][SUB]        = get_update_liz_divergent();

    try {
        return funcs.at(conn->type).at(conn->opcode);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Unimplemented connection type!");
    }
}
