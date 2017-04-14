#include "state/spiking_attributes.h"

SpikingAttributes::SpikingAttributes(LayerList &layers, Kernel<ATTRIBUTE_ARGS> kernel)
        : Attributes(layers, BIT, kernel) {
    this->voltage = Pointer<float>(total_neurons);
    Attributes::register_variable(&this->voltage);
}

void SpikingAttributes::process_weight_matrix(WeightMatrix* matrix) {
    Connection *conn = matrix->connection;
    Pointer<float> mData = matrix->get_data();

    int num_weights = conn->get_num_weights();

    // Baseline
    transfer_weights(mData, mData + num_weights, num_weights);

    // Trace
    clear_weights(mData + 2*num_weights, num_weights);
}

/******************************************************************************/
/************************* TRACE ACTIVATOR KERNELS ****************************/
/******************************************************************************/

#define EXTRACT_TRACES \
    float *traces = weights + (2*num_weights);

#define CALC_VAL(from_index, weight_index) \
    float val = extractor(outputs[from_index], delay); \
    float trace = traces[weight_index]; \
\
    traces[weight_index] = trace = ((val > 0.0) \
        ? g : (trace - (trace / tau))); \
    val = weights[weight_index] * trace * adjusted_voltage;

#define GET_DEST_VOLTAGE(to_index) \
    float voltage = *att->voltage.get(to_index); \
    float adjusted_voltage = 1; \
    switch(opcode) { \
        case(AMPA): \
            adjusted_voltage = voltage; \
            break; \
        case(GABAA): \
            adjusted_voltage = voltage + 70; \
            break; \
        case(NMDA): {\
            float temp = (voltage + 80) / 60; \
            temp = temp * temp; \
            adjusted_voltage = (temp / (1+temp)) * voltage; \
            break; \
        } \
        case(GABAB): \
            adjusted_voltage = voltage + 90; \
            break; \
    }

#define ACTIV_EXTRACTIONS \
    SpikingAttributes *att = (SpikingAttributes*)synapse_data.to_attributes; \
    float tau = 10; \
    float g = 1; \
    switch(opcode) { \
        case(AMPA): \
            tau = 5; \
            g = -0.0025; \
            break; \
        case(GABAA): \
            tau = 7; \
            /* g = -0.005; */ \
            g = -0.0025; \
            break; \
        case(NMDA): \
            tau = 150; \
            g = -0.000025; \
            break; \
        case(GABAB): \
            tau = 150; \
            g = -0.00005; \
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

    if (convolutional) {
        CALC_VAL(from_index, (to_index*num_weights + weight_index));
        sum += val;
    } else {
        CALC_VAL(from_index, weight_index);
        sum += val;
    },

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
    if (convolutional) {
        UPDATE_TRACE((to_index*num_weights + weight_index));
    } else {
        UPDATE_TRACE(weight_index);
    }
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
        case CONVOLUTIONAL:
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

#define EXTRACT_BASELINES \
    float *baselines = weights + num_weights;

#define UPDATE_WEIGHT(weight_index, dest_spike) \
    float trace = (dest_spike) ? traces[weight_index] : 0.0; \
    weights[weight_index] = \
        MAX(baselines[weight_index], \
            MIN(max_weight, \
                (weights[weight_index] * decay) \
                + (-trace * coefficient)));

#define UPDATE_WEIGHT_CONVOLUTIONAL(weight_index, dest_spike) \
    float trace = 0.0; \
    if (dest_spike) { \
        for (int i = 0; i < to_size; ++i) { \
            trace += traces[i*num_weights + weight_index]; \
        } \
        trace /= to_size; \
    } \
    weights[weight_index] = \
        MAX(baselines[weight_index], \
            MIN(max_weight, \
                (weights[weight_index] * decay) \
                + (-trace * coefficient)));

#define UPDATE_EXTRACTIONS \
    EXTRACT_TRACES; \
    EXTRACT_BASELINES; \
    float decay = 0.9999; \
    float coefficient = 10; \
    switch(opcode) { \
        case(AMPA): \
            decay = 0.999; \
            coefficient = 40; \
            break; \
        case(GABAA): \
            decay = 0.999; \
            coefficient = 40; \
            break; \
        case(NMDA): \
            decay = 1.0; \
            coefficient = 10; \
            break; \
        case(GABAB): \
            decay = 1.0; \
            coefficient = 10; \
            break; \
    }

#define GET_DEST_ACTIVITY(to_index) \
    bool dest_spike = extractor(destination_outputs[to_index], 0);

CALC_FULLY_CONNECTED(update_fully_connected_trace,
    UPDATE_EXTRACTIONS;,
    GET_DEST_ACTIVITY(to_index);,
    UPDATE_WEIGHT(weight_index, dest_spike),
    ; );
CALC_ONE_TO_ONE(update_one_to_one_trace,
    UPDATE_EXTRACTIONS;,
    GET_DEST_ACTIVITY(index);
    UPDATE_WEIGHT(index, dest_spike)
    );
CALC_CONVERGENT(update_convergent_trace,
    UPDATE_EXTRACTIONS;,
    GET_DEST_ACTIVITY(to_index);,
    UPDATE_WEIGHT(weight_index, dest_spike),
    ; );
CALC_ONE_TO_ONE(update_convolutional_trace,
    UPDATE_EXTRACTIONS;,
    GET_DEST_ACTIVITY(index);
    UPDATE_WEIGHT_CONVOLUTIONAL(index, dest_spike)
    );
CALC_DIVERGENT(update_divergent_trace,
    UPDATE_EXTRACTIONS;,
    GET_DEST_ACTIVITY(to_index);,
    UPDATE_WEIGHT(weight_index, dest_spike),
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
        case CONVOLUTIONAL:
            return get_update_convolutional_trace();
        case DIVERGENT:
            return get_update_divergent_trace();
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}
