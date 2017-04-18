#include <math.h>

#include "state/spiking_attributes.h"

SpikingAttributes::SpikingAttributes(LayerList &layers)
        : Attributes(layers, BIT) {
    this->voltage = Pointer<float>(total_neurons);
    this->neuron_trace = Pointer<float>(total_neurons);
    Attributes::register_variable(&this->voltage);
    Attributes::register_variable(&this->neuron_trace);
}

void SpikingAttributes::process_weight_matrix(WeightMatrix* matrix) {
    Connection *conn = matrix->connection;
    Pointer<float> mData = matrix->get_data();

    int num_weights = conn->get_num_weights();

    // Copy initial weights to baseline
    transfer_weights(mData, mData + (1*num_weights), num_weights);

    // Traces
    for (int i = 2 ; i < this->get_matrix_depth(conn) ; ++i) {
        clear_weights(mData + i*num_weights, num_weights);
    }
}

/******************************************************************************/
/************************* TRACE ACTIVATOR KERNELS ****************************/
/******************************************************************************/

#define AMPA_TAU    0.8    // tau = 5
#define GABAA_TAU   0.857  // tau = 7
#define NMDA_TAU    0.993  // tau = 150
#define GABAB_TAU   0.993  // tau = 150
#define PLASTIC_TAU 0.95   // tau = 20

#define SHORT_G  -0.0075   // -0.0025 * 3
#define LONG_G   -0.00075  // -0.00025 * 3
#define PLASTIC_BASE 0.1

#define EXTRACT_TRACES \
    float *baselines    = weights + (1*num_weights); \
    float *short_traces = weights + (2*num_weights); \
    float *long_traces  = weights + (3*num_weights); \
    float *pl_traces    = weights + (4*num_weights);

#define CALC_VAL(from_index, weight_index) \
    float val = 0.0; \
    if (baselines[weight_index] > 0.0) { \
        bool spike = extractor(outputs[from_index], delay) > 0.0; \
    \
        if (opcode == ADD or opcode == SUB) { \
            float trace = short_traces[weight_index]; \
            short_traces[weight_index] = trace = (spike \
                ? SHORT_G : (trace * short_tau)); \
            val += trace * short_adjusted_voltage \
                * weights[weight_index]; \
        \
            trace = long_traces[weight_index]; \
            long_traces[weight_index] = trace = (spike \
                ? LONG_G : (trace * long_tau)); \
            val += trace * long_adjusted_voltage \
                * weights[weight_index]; \
        \
            trace = pl_traces[weight_index]; \
            pl_traces[weight_index] = (spike \
                ? PLASTIC_BASE : (trace * PLASTIC_TAU)); \
        } else { \
            val = spike * weights[weight_index]; \
        } \
    }

#define GET_DEST_VOLTAGE(to_index) \
    float voltage = *att->voltage.get(to_index); \
    float short_adjusted_voltage = 1; \
    float long_adjusted_voltage = 1; \
    switch(opcode) { \
        case(ADD): {\
            short_adjusted_voltage = voltage; \
            float temp = pow((voltage + 80) / 60, 2); \
            long_adjusted_voltage = (temp / (1+temp)) * voltage; \
            break; \
        } \
        case(SUB): \
            short_adjusted_voltage = voltage + 70; \
            long_adjusted_voltage = voltage + 90; \
            break; \
    }

#define ACTIV_EXTRACTIONS \
    SpikingAttributes *att = \
        (SpikingAttributes*)synapse_data.to_attributes; \
    float short_tau = 0.9; \
    float long_tau = 0.9; \
    switch(opcode) { \
        case(ADD): \
            short_tau = AMPA_TAU; \
            long_tau  = NMDA_TAU; \
            break; \
        case(SUB): \
            short_tau = GABAA_TAU; \
            long_tau  = GABAB_TAU; \
            break; \
    } \
    EXTRACT_TRACES;

#define AGGREGATE(to_index, sum) \
    inputs[to_index] = inputs[to_index] + sum;


/* Trace versions of activator functions */
CALC_FULLY_CONNECTED(activate_fully_connected_trace,
    ACTIV_EXTRACTIONS,

    GET_DEST_VOLTAGE(to_index)
    float sum = 0.0;,

    CALC_VAL(from_index, weight_index)
    sum += val;,

    AGGREGATE(to_index, sum));

CALC_ONE_TO_ONE(activate_one_to_one_trace,
    ACTIV_EXTRACTIONS,

    GET_DEST_VOLTAGE(index)
    CALC_VAL(index, index)
    AGGREGATE(index, inputs[index]));
CALC_CONVERGENT(activate_convergent_trace,
    ACTIV_EXTRACTIONS,

    GET_DEST_VOLTAGE(to_index)
    float sum = 0.0;,

    CALC_VAL(from_index, weight_index);
    sum += val;,

    AGGREGATE(to_index, sum);
);
CALC_DIVERGENT(activate_divergent_trace,
    ACTIV_EXTRACTIONS,

    GET_DEST_VOLTAGE(to_index)
    float sum = 0.0;,

    CALC_VAL(from_index, weight_index)
    sum += val;,

    AGGREGATE(to_index, sum);
);

#define UPDATE_TRACE(weight_index)

/* Second order */
ACTIVATE_FULLY_CONNECTED_SECOND_ORDER(
        activate_fully_connected_trace_second_order,
    ACTIV_EXTRACTIONS,
    UPDATE_TRACE(weight_index));
ACTIVATE_ONE_TO_ONE_SECOND_ORDER(
        activate_one_to_one_trace_second_order,
    ACTIV_EXTRACTIONS,
    UPDATE_TRACE(index));
ACTIVATE_CONVERGENT_SECOND_ORDER(
        activate_convergent_trace_second_order,
    ACTIV_EXTRACTIONS,
    UPDATE_TRACE(weight_index);
);
ACTIVATE_DIVERGENT_SECOND_ORDER(
        activate_divergent_trace_second_order,
    ACTIV_EXTRACTIONS,
    UPDATE_TRACE(weight_index);
);

Kernel<SYNAPSE_ARGS> SpikingAttributes::get_activator(
        ConnectionType type, bool second_order) {
    switch (type) {
        case FULLY_CONNECTED:
            return (second_order)
                ? get_activate_fully_connected_trace_second_order()
                : get_activate_fully_connected_trace();
        case ONE_TO_ONE:
            return (second_order)
                ? get_activate_one_to_one_trace_second_order()
                : get_activate_one_to_one_trace();
        case CONVERGENT:
            return (second_order)
                ? get_activate_convergent_trace_second_order()
                : get_activate_convergent_trace();
        case DIVERGENT:
            return (second_order)
                ? get_activate_divergent_trace_second_order()
                : get_activate_divergent_trace();
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}

/******************************************************************************/
/************************** TRACE UPDATER KERNELS *****************************/
/******************************************************************************/

#define DECAY 1.0
#define LEARNING_RATE 0.1
#define MIN_WEIGHT 0.0

#define UPDATE_WEIGHT(weight_index, from_index, dest_trace, dest_spike) \
    bool src_spike = extractor(outputs[from_index], 0); \
    float src_trace = pl_traces[weight_index]; \
    float weight = weights[weight_index]; \
    if (baselines[weight_index] > 0.0) { \
        float delta = (src_spike) \
            ? ((max_weight - weight) * src_trace * LEARNING_RATE) : 0.0; \
        delta -= (dest_spike) \
            ? ((weight - MIN_WEIGHT) * dest_trace * LEARNING_RATE) : 0.0; \
        weights[weight_index] = (weight * DECAY) + delta; \
    }

#define UPDATE_EXTRACTIONS \
    float *baselines = weights + (1*num_weights); \
    float *pl_traces = weights + (4*num_weights); \
    float *neuron_traces = \
        ((SpikingAttributes*)synapse_data.to_attributes) \
            ->neuron_trace.get();

#define GET_DEST_ACTIVITY(to_index) \
    float dest_trace = neuron_traces[to_index]; \
    bool dest_spike = extractor(destination_outputs[to_index], 0);

CALC_FULLY_CONNECTED(update_fully_connected_trace,
    UPDATE_EXTRACTIONS;,
    GET_DEST_ACTIVITY(to_index);,
    UPDATE_WEIGHT(weight_index, from_index, dest_trace, dest_spike),
    ; );
CALC_ONE_TO_ONE(update_one_to_one_trace,
    UPDATE_EXTRACTIONS;,
    GET_DEST_ACTIVITY(index);
    UPDATE_WEIGHT(index, index, dest_trace, dest_spike)
    );
CALC_CONVERGENT(update_convergent_trace,
    UPDATE_EXTRACTIONS;,
    GET_DEST_ACTIVITY(to_index);,
    UPDATE_WEIGHT(weight_index, from_index, dest_trace, dest_spike),
    ; );
CALC_DIVERGENT(update_divergent_trace,
    UPDATE_EXTRACTIONS;,
    GET_DEST_ACTIVITY(to_index);,
    UPDATE_WEIGHT(weight_index, from_index, dest_trace, dest_spike),
    ; );

Kernel<SYNAPSE_ARGS> SpikingAttributes::get_updater(
        ConnectionType conn_type, bool second_order) {
    switch (conn_type) {
        case FULLY_CONNECTED:
            return get_update_fully_connected_trace();
        case ONE_TO_ONE:
            return get_update_one_to_one_trace();
        case CONVERGENT:
            return get_update_convergent_trace();
        case DIVERGENT:
            return get_update_divergent_trace();
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}
