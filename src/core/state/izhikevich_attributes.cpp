#include <string>
#include <math.h>

#include "state/izhikevich_attributes.h"
#include "util/tools.h"

REGISTER_ATTRIBUTES(IzhikevichAttributes, "izhikevich")

/******************************************************************************/
/******************************** PARAMS **************************************/
/******************************************************************************/

#define DEF_PARAM(name, a,b,c,d) \
    static const IzhikevichParameters name = IzhikevichParameters(a,b,c,d);

/* Izhikevich Parameters Table */
DEF_PARAM(DEFAULT          , 0.02, 0.2 , -70.0, 2   ); // Default
DEF_PARAM(REGULAR          , 0.02, 0.2 , -65.0, 8   ); // Regular Spiking
DEF_PARAM(BURSTING         , 0.02, 0.2 , -55.0, 4   ); // Intrinsically Bursting
DEF_PARAM(CHATTERING       , 0.02, 0.2 , -50.0, 2   ); // Chattering
DEF_PARAM(FAST             , 0.1 , 0.2 , -65.0, 2   ); // Fast Spiking
DEF_PARAM(LOW_THRESHOLD    , 0.02, 0.25, -65.0, 2   ); // Low Threshold
DEF_PARAM(THALAMO_CORTICAL , 0.02, 0.25, -65.0, 0.05); // Thalamo-cortical
DEF_PARAM(RESONATOR        , 0.1 , 0.26, -65.0, 2   ); // Resonator

static IzhikevichParameters create_parameters(std::string str) {
    if (str == "random positive") {
        // (ai; bi) = (0:02; 0:2) and (ci; di) = (-65; 8) + (15;-6)r2
        float a = 0.02;
        float b = 0.2; // increase for higher frequency oscillations

        float rand = fRand(0, 1);
        float c = -65.0 + (15.0 * rand * rand);

        rand = fRand(0, 1);
        float d = 8.0 - (6.0 * (rand * rand));
        return IzhikevichParameters(a,b,c,d);
    } else if (str == "random negative") {
        //(ai; bi) = (0:02; 0:25) + (0:08;-0:05)ri and (ci; di)=(-65; 2).
        float rand = fRand(0, 1);
        float a = 0.02 + (0.08 * rand);

        rand = fRand(0, 1);
        float b = 0.25 - (0.05 * rand);

        float c = -65.0;
        float d = 2.0;
        return IzhikevichParameters(a,b,c,d);
    }
    else if (str == "default")            return DEFAULT;
    else if (str == "regular")            return REGULAR;
    else if (str == "bursting")           return BURSTING;
    else if (str == "chattering")         return CHATTERING;
    else if (str == "fast")               return FAST;
    else if (str == "low_threshold")      return LOW_THRESHOLD;
    else if (str == "thalamo_cortical")   return THALAMO_CORTICAL;
    else if (str == "resonator")          return RESONATOR;
    else
        ErrorManager::get_instance()->log_error(
            "Unrecognized parameter string: " + str);
}

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

/* Voltage threshold for neuron spiking. */
#define IZ_SPIKE_THRESH 30

/* Euler resolution for voltage update. */
#define IZ_EULER_RES 2

/* Milliseconds per timestep */
#define IZ_TIMESTEP_MS 1

#define TRACE_BASE 0.1
#define TRACE_TAU 0.98  // 50ms

BUILD_ATTRIBUTE_KERNEL(IzhikevichAttributes, iz_attribute_kernel,
    IzhikevichAttributes *iz_att = (IzhikevichAttributes*)att;

    float *ampa_conductances = iz_att->ampa_conductance.get(other_start_index);
    float *nmda_conductances = iz_att->nmda_conductance.get(other_start_index);
    float *gabaa_conductances = iz_att->gabaa_conductance.get(other_start_index);
    float *gabab_conductances = iz_att->gabab_conductance.get(other_start_index);
    float *multiplicative_factors = iz_att->multiplicative_factor.get(other_start_index);

    float *voltages = iz_att->voltage.get(other_start_index);
    float *recoveries = iz_att->recovery.get(other_start_index);
    float *neuron_traces = iz_att->neuron_trace.get(other_start_index);
    unsigned int *spikes = (unsigned int*)outputs;
    IzhikevichParameters *params = iz_att->neuron_parameters.get(other_start_index);

    ,

    /**********************
     *** VOLTAGE UPDATE ***
     **********************/
    float ampa_conductance = ampa_conductances[nid];
    float nmda_conductance = nmda_conductances[nid];
    float gabaa_conductance = gabaa_conductances[nid];
    float gabab_conductance = gabab_conductances[nid];
    float multiplicative_factor = multiplicative_factors[nid];

    float voltage = voltages[nid];
    float recovery = recoveries[nid];
    float base_current = inputs[nid];

    float a = params[nid].a;
    float b = params[nid].b;

    // Euler's method for voltage/recovery update
    // If the voltage exceeds the spiking threshold, break
    for (int i = 0 ; i < IZ_TIMESTEP_MS * IZ_EULER_RES && voltage < IZ_SPIKE_THRESH ; ++i) {
        float current = base_current;
        current += ampa_conductance * voltage;
        float temp = pow((voltage + 80) / 60, 2);
        current += nmda_conductance * (temp / (1+temp)) * voltage;
        current += gabaa_conductance * (voltage + 70);
        current += gabab_conductance * (voltage + 90);
        current *= (1+multiplicative_factor);

        //if (current != 0.0) printf("%f\n", current);

        float delta_v = (0.04 * voltage * voltage) +
                        (5*voltage) + 140 - recovery + current;
        voltage += delta_v / IZ_EULER_RES;
        recovery += a * ((b * voltage) - recovery) / IZ_EULER_RES;
    }

    ampa_conductances[nid] = 0.0;
    nmda_conductances[nid] = 0.0;
    gabaa_conductances[nid] = 0.0;
    gabab_conductances[nid] = 0.0;
    multiplicative_factors[nid] = 0.0;

    /********************
     *** SPIKE UPDATE ***
     ********************/
    // Determine if spike occurred
    unsigned int spike = voltage >= IZ_SPIKE_THRESH;

    // Reduce reads, chain values.
    unsigned int next_value = spikes[nid];

    // Shift all the bits.
    // Check if next word is odd (1 for LSB).
    int index;
    for (index = 0 ; index < history_size-1 ; ++index) {
        unsigned int curr_value = next_value;
        next_value = spikes[size * (index + 1) + nid];

        // Shift bits, carry over LSB from next value.
        spikes[size*index + nid] = (curr_value >> 1) | (next_value << 31);
    }

    // Least significant value already loaded into next_value.
    // Index moved appropriately from loop.
    spikes[size*index + nid] = (next_value >> 1) | (spike << 31);

    // Update trace, voltage, recovery
    neuron_traces[nid] = (spike)
        ? TRACE_BASE : (neuron_traces[nid] * TRACE_TAU);
    voltages[nid] = (spike) ? params[nid].c : voltage;
    recoveries[nid] = recovery + ((spike) ? params[nid].d : 0.0);
)

/******************************************************************************/
/************************* TRACE ACTIVATOR KERNELS ****************************/
/******************************************************************************/

#define AMPA_TAU    0.8    // tau = 5
#define GABAA_TAU   0.857  // tau = 7
#define NMDA_TAU    0.993  // tau = 150
#define GABAB_TAU   0.993  // tau = 150
#define PLASTIC_TAU 0.95   // tau = 20

#define SHORT_G  -0.0025
#define LONG_G   -0.000025
#define PLASTIC_BASE 0.1

#define EXTRACT_TRACES \
    float *baselines    = weights + (1*num_weights); \
    float *short_traces = weights + (2*num_weights); \
    float *long_traces  = weights + (3*num_weights); \
    float *pl_traces    = weights + (4*num_weights);

#define CALC_VAL(from_index, weight_index) \
    if (baselines[weight_index] > 0.0) { \
        bool spike = extractor(outputs[from_index], delay) > 0.0; \
    \
        if (opcode == ADD or opcode == SUB) { \
            float trace = short_traces[weight_index]; \
            short_traces[weight_index] = trace = (spike \
                ? SHORT_G : (trace * short_tau)); \
            short_sum += trace * weights[weight_index]; \
        \
            trace = long_traces[weight_index]; \
            long_traces[weight_index] = trace = (spike \
                ? LONG_G : (trace * long_tau)); \
            long_sum += trace * weights[weight_index]; \
        \
            trace = pl_traces[weight_index]; \
            pl_traces[weight_index] = (spike \
                ? PLASTIC_BASE : (trace * PLASTIC_TAU)); \
        } else { \
            short_sum += spike * weights[weight_index]; \
        } \
    }

#define ACTIV_EXTRACTIONS \
    IzhikevichAttributes *att = \
        (IzhikevichAttributes*)synapse_data.to_attributes; \
    int to_start_index = synapse_data.to_start_index; \
    float *short_conductances = nullptr; \
    float *long_conductances = nullptr; \
\
    float short_tau = 0.9; \
    float long_tau = 0.9; \
    switch(opcode) { \
        case(ADD): \
            short_conductances = att->ampa_conductance.get(to_start_index); \
            long_conductances = att->nmda_conductance.get(to_start_index); \
            short_tau = AMPA_TAU; \
            long_tau  = NMDA_TAU; \
            break; \
        case(SUB): \
            short_conductances = att->gabaa_conductance.get(to_start_index); \
            long_conductances = att->gabab_conductance.get(to_start_index); \
            short_tau = GABAA_TAU; \
            long_tau  = GABAB_TAU; \
            break; \
        case(MULT): \
            short_conductances = att->multiplicative_factor.get(to_start_index); \
            break; \
    } \
    EXTRACT_TRACES;

#define AGGREGATE(to_index) \
    short_conductances[to_index] += short_sum; \
    if (long_conductances != nullptr) long_conductances[to_index] += long_sum;


/* Trace versions of activator functions */
CALC_FULLY_CONNECTED(activate_fully_connected_trace,
    ACTIV_EXTRACTIONS,

    float short_sum = 0.0;
    float long_sum = 0.0;,

    CALC_VAL(from_index, weight_index),

    AGGREGATE(to_index)
);
CALC_ONE_TO_ONE(activate_one_to_one_trace,
    ACTIV_EXTRACTIONS,

    float short_sum = 0.0;
    float long_sum = 0.0;
    CALC_VAL(index, index)
    AGGREGATE(index)
);
CALC_CONVERGENT(activate_convergent_trace,
    ACTIV_EXTRACTIONS,

    float short_sum = 0.0;
    float long_sum = 0.0;,

    CALC_VAL(from_index, weight_index),

    AGGREGATE(to_index);
);
CALC_DIVERGENT(activate_divergent_trace,
    ACTIV_EXTRACTIONS,

    float short_sum = 0.0;
    float long_sum = 0.0;,

    CALC_VAL(from_index, weight_index),

    AGGREGATE(to_index);
);

Kernel<SYNAPSE_ARGS> IzhikevichAttributes::get_activator(
        ConnectionType type, bool second_order) {
    switch (type) {
        case FULLY_CONNECTED:
            return get_activate_fully_connected_trace();
        case ONE_TO_ONE:
            return get_activate_one_to_one_trace();
        case CONVERGENT:
            return get_activate_convergent_trace();
        case DIVERGENT:
            return get_activate_divergent_trace();
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}

/******************************************************************************/
/************************** TRACE UPDATER KERNELS *****************************/
/******************************************************************************/

#define DECAY 1.0
#define LEARNING_RATE 1.0
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
    int to_start_index = synapse_data.to_start_index; \
    float *neuron_traces = \
        ((IzhikevichAttributes*)synapse_data.to_attributes) \
            ->neuron_trace.get(to_start_index);

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

Kernel<SYNAPSE_ARGS> IzhikevichAttributes::get_updater(
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

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

IzhikevichAttributes::IzhikevichAttributes(LayerList &layers)
        : Attributes(layers, BIT) {
    // Condutances
    this->ampa_conductance = Pointer<float>(total_neurons);
    this->nmda_conductance = Pointer<float>(total_neurons);
    this->gabaa_conductance = Pointer<float>(total_neurons);
    this->gabab_conductance = Pointer<float>(total_neurons);
    this->multiplicative_factor = Pointer<float>(total_neurons);
    Attributes::register_variable(&this->ampa_conductance);
    Attributes::register_variable(&this->nmda_conductance);
    Attributes::register_variable(&this->gabaa_conductance);
    Attributes::register_variable(&this->gabab_conductance);
    Attributes::register_variable(&this->multiplicative_factor);

    // Neuron variables
    this->voltage = Pointer<float>(total_neurons);
    this->recovery = Pointer<float>(total_neurons);
    this->neuron_trace = Pointer<float>(total_neurons);
    Attributes::register_variable(&this->recovery);
    Attributes::register_variable(&this->neuron_parameters);
    Attributes::register_variable(&this->voltage);

    // Neuron parameters
    this->neuron_parameters = Pointer<IzhikevichParameters>(total_neurons);
    Attributes::register_variable(&this->neuron_trace);

    // Fill in table
    int start_index = 0;
    for (auto& layer : layers) {
        std::string init_param;
        try {
            init_param = layer->config->get_property("init");
        } catch (...) {
            ErrorManager::get_instance()->log_warning(
                "Unspecified Izhikevich init params for layer \""
                + layer->name + "\" -- using regular spiking.");
            init_param = "regular";
        }
        IzhikevichParameters params = create_parameters(init_param);
        for (int j = 0 ; j < layer->size ; ++j) {
            neuron_parameters[start_index+j] = params;
            voltage[start_index+j] = params.c;
            recovery[start_index+j] = params.b * params.c;
        }
        start_index += layer->size;
    }
}

void IzhikevichAttributes::process_weight_matrix(WeightMatrix* matrix) {
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
