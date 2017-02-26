#include "state/spiking_attributes.h"
#include "util/tools.h"
#include "util/error_manager.h"
#include "util/parallel.h"

SpikingAttributes::SpikingAttributes(Structure* structure, ATTRIBUTE_KERNEL kernel)
        : Attributes(structure, BIT, kernel) {
    this->voltage = Pointer<float>(total_neurons);
    this->current = this->input;
    this->spikes = this->output.cast<unsigned int>();
}

SpikingAttributes::~SpikingAttributes() {
    this->voltage.free();
}

void SpikingAttributes::transfer_to_device() {
    Attributes::transfer_to_device();

    this->current = this->input;
    this->spikes = this->output.cast<unsigned int>();
    this->voltage.transfer_to_device();
}

void SpikingAttributes::process_weight_matrix(WeightMatrix* matrix) {
    Connection *conn = matrix->connection;
    Pointer<float> mData = matrix->get_data();
    if (conn->plastic) {
        int num_weights = conn->get_num_weights();

        // Baseline
        transfer_weights(mData, mData + num_weights, num_weights);

        // Trace
        clear_weights(mData + 2*num_weights, num_weights);
    }
}

/******************************************************************************/
/************************* TRACE ACTIVATOR KERNELS ****************************/
/******************************************************************************/

// Different minimum functions are used on the host and device
#ifdef PARALLEL
#define MIN min
#else
#include <algorithm>
#define MIN std::fmin
#endif

#define MOD_RATE 0.05
#define MOD_DECAY 0.005
#define MOD_MAX 10.0

#define EXTRACT_BASELINES \
    float *baselines = weights + num_weights;

#define EXTRACT_TRACES \
    float *traces = weights + (2*num_weights);

#define UPDATE_TRACE(weight_index) \
    if (plastic) { \
        float old_trace = traces[weight_index]; \
        float new_trace = old_trace + (MOD_RATE * val) - (MOD_DECAY * old_trace); \
        traces[weight_index] = MIN(MOD_MAX, new_trace);  \
    }

/* Trace versions of activator functions */
ACTIVATE_FULLY_CONNECTED(activate_fully_connected_trace,
    EXTRACT_TRACES,
    UPDATE_TRACE(weight_index));
ACTIVATE_ONE_TO_ONE(activate_one_to_one_trace,
    EXTRACT_TRACES,
    UPDATE_TRACE(index));
ACTIVATE_CONVERGENT(activate_convergent_trace,
    EXTRACT_TRACES,
    if (convolutional) {
        UPDATE_TRACE((to_index*num_weights + weight_index));
    } else {
        UPDATE_TRACE(weight_index);
    }
);

SYNAPSE_KERNEL SpikingAttributes::get_activator(ConnectionType type) {
    switch (type) {
        case FULLY_CONNECTED:
            return activate_fully_connected_trace;
        case ONE_TO_ONE:
            return activate_one_to_one_trace;
        case CONVERGENT:
        case CONVOLUTIONAL:
            return activate_convergent_trace;
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}

/******************************************************************************/
/************************** TRACE UPDATER KERNELS *****************************/
/******************************************************************************/

#define SUM_COEFFICIENT 0.01
#define WEIGHT_DECAY 0.0001

#define UPDATE_WEIGHT(weight_index, input) \
    float old_weight = weights[weight_index]; \
    float new_weight = old_weight + (traces[weight_index] * input * SUM_COEFFICIENT) \
                        - (WEIGHT_DECAY * (old_weight - baselines[weight_index])); \
    weights[weight_index] = (new_weight > max_weight) ? max_weight : new_weight;

#define UPDATE_WEIGHT_CONVOLUTIONAL(weight_index, input) \
    float old_weight = weights[weight_index]; \
    float t = 0.0; \
    for (int i = 0; i < to_size; ++i) { \
        t += traces[i*num_weights + weight_index]; \
    } \
    t /= to_size; \
    float new_weight = old_weight + (t * input * SUM_COEFFICIENT) \
                        - (WEIGHT_DECAY * (old_weight - baselines[weight_index])); \
    weights[weight_index] = (new_weight > max_weight) ? max_weight : new_weight;

CALC_FULLY_CONNECTED(update_fully_connected_trace,
    EXTRACT_TRACES;
    EXTRACT_BASELINES;,
    float sum = inputs[to_index];,
    UPDATE_WEIGHT(weight_index, sum),
    ; );
CALC_ONE_TO_ONE(update_one_to_one_trace,
    EXTRACT_TRACES;
    EXTRACT_BASELINES;,
    UPDATE_WEIGHT(index, inputs[index]));
CALC_CONVERGENT(update_convergent_trace,
    EXTRACT_TRACES;
    EXTRACT_BASELINES;,
    float sum = inputs[to_index];,
    UPDATE_WEIGHT(weight_index, sum),
    ; );
CALC_ONE_TO_ONE(update_convolutional_trace,
    EXTRACT_TRACES;
    EXTRACT_BASELINES;,
    UPDATE_WEIGHT_CONVOLUTIONAL(index, inputs[index]);
    );

SYNAPSE_KERNEL SpikingAttributes::get_updater(ConnectionType conn_type) {
    switch (conn_type) {
        case FULLY_CONNECTED:
            return update_fully_connected_trace;
        case ONE_TO_ONE:
            return update_one_to_one_trace;
        case CONVERGENT:
            return update_convergent_trace;
        case CONVOLUTIONAL:
            return update_convolutional_trace;
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}
