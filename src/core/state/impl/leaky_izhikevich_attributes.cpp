#include <string>
#include <math.h>

#include "state/impl/leaky_izhikevich_attributes.h"
#include "state/weight_matrix.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/tools.h"

REGISTER_ATTRIBUTES(LeakyIzhikevichAttributes, "leaky_izhikevich", BIT)

/******************************************************************************/
/************************** TRACE UPDATER KERNELS *****************************/
/******************************************************************************/

/* STDP A constant */
#define STDP_A 0.004

#define PLASTIC_TAU       0.95   // tau = 20

#define UPDATE_EXTRACTIONS \
    float *presyn_traces = weights + (3*num_weights); \
    float *eligibilities = weights + (6*num_weights); \
    int   *delays        = (int*)weights + (7*num_weights); \
\
    LeakyIzhikevichAttributes *att = \
        (LeakyIzhikevichAttributes*)synapse_data.attributes; \
    float *to_traces = att->postsyn_trace.get(synapse_data.to_start_index); \
    float learning_rate = \
        att->learning_rate.get()[synapse_data.connection_index];

#define GET_DEST_ACTIVITY \
    float dest_trace = to_traces[to_index]; \
    float dest_spike = extractor(destination_outputs[to_index], 0);

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
        float src_spike = extractor(outputs[from_index], delays[weight_index]); \
    \
        /* Update presynaptic trace */ \
        float src_trace = \
            presyn_traces[weight_index] = \
                (src_spike > 0.0) \
                    ? STDP_A \
                    : (presyn_traces[weight_index] * PLASTIC_TAU); \
    \
        /* Compute delta from short term dynamics */ \
        float c_delta = \
                (dest_spike * src_trace) \
                - (src_spike  * dest_trace); \
\
        float c = eligibilities[weight_index] + (learning_rate * c_delta); \
        /* if (weight_index == 500) printf("(%2.12f  %2.12f  %2.12f)\n", c, c-(c-C_A)*C_TAU, weight+c); */ \
\
        /* Update eligibility trace */ \
        eligibilities[weight_index] -= (c - C_A) * C_TAU; \
\
        /* Calculate new weight */ \
        weight += c; \
        /* if (to_index == 0 and (c > 100*C_A or c < -C_A)) printf("(%f %f)  ", c, weight); */ \
\
        /* Ensure weight stays within boundaries */ \
        weights[weight_index] = weight = \
            MAX(MIN_WEIGHT, MIN(max_weight, weight)); \
\
        /*
        float change = learning_rate * weight_delta; \
        if (change > 0.05 or change < -0.05) \
            printf("(%8d w=%7.5f delt=%8.5f c=%9.4f rew=%7.4f change=%8.5f)\n", \
                weight_index, weight, weight_delta, c, dopamine, change); */ \
    }

CALC_ALL(update_liz_add,
    UPDATE_EXTRACTIONS,
    GET_DEST_ACTIVITY,
    UPDATE_WEIGHT,
; );

Kernel<SYNAPSE_ARGS> LeakyIzhikevichAttributes::get_updater(Connection *conn) {
    if (conn->second_order)
        ErrorManager::get_instance()->log_error(
            "Unimplemented connection type!");

    std::map<ConnectionType, std::map<Opcode, Kernel<SYNAPSE_ARGS> > > funcs;
    funcs[FULLY_CONNECTED][ADD]  = get_update_liz_add_fully_connected();
    funcs[SUBSET][ADD]           = get_update_liz_add_subset();
    funcs[ONE_TO_ONE][ADD]       = get_update_liz_add_one_to_one();
    funcs[CONVERGENT][ADD]       = get_update_liz_add_convergent();
    funcs[DIVERGENT][ADD]        = get_update_liz_add_divergent();

    try {
        return funcs.at(conn->type).at(conn->opcode);
    } catch (std::out_of_range) {
        ErrorManager::get_instance()->log_error(
            "Unimplemented connection type!");
    }
}
