#include <string>
#include <math.h>

#include "state/izhikevich_attributes.h"
#include "engine/kernel/synapse_kernel.h"
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
#define IZ_EULER_RES 10

/* Milliseconds per timestep */
#define IZ_TIMESTEP_MS 1

/* Time dynamics of postsynaptic spikes */
#define TRACE_TAU 0.95  // 20ms

BUILD_ATTRIBUTE_KERNEL(IzhikevichAttributes, iz_attribute_kernel,
    IzhikevichAttributes *iz_att = (IzhikevichAttributes*)att;

    float *ampa_conductances = iz_att->ampa_conductance.get(other_start_index);
    float *nmda_conductances = iz_att->nmda_conductance.get(other_start_index);
    float *gabaa_conductances = iz_att->gabaa_conductance.get(other_start_index);
    float *gabab_conductances = iz_att->gabab_conductance.get(other_start_index);
    float *multiplicative_factors = iz_att->multiplicative_factor.get(other_start_index);

    float *voltages = iz_att->voltage.get(other_start_index);
    float *recoveries = iz_att->recovery.get(other_start_index);
    float *postsyn_traces = iz_att->postsyn_trace.get(other_start_index);
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
    for (int i = 0 ; (i < IZ_TIMESTEP_MS * IZ_EULER_RES)
            and (voltage < IZ_SPIKE_THRESH) ; ++i) {
        // Start with AMPA conductance
        float current = -ampa_conductance * voltage;

        // NMDA nonlinear voltage dependence
        float temp = powf((voltage + 80) / 60, 2);
        current -= nmda_conductance * (temp / (1+temp)) * voltage;

        // GABA conductances
        current -= gabaa_conductance * (voltage + 70);
        current -= gabab_conductance * (voltage + 90);

        // Multiplicative factor for synaptic currents
        current *= 1 + multiplicative_factor;

        // Add the base current after multiplicative factor
        current += base_current;

        // Update voltage
        float delta_v = (0.04 * voltage * voltage) +
                        (5*voltage) + 140 - recovery + current;
        voltage += delta_v / IZ_EULER_RES;

        // If the voltage explodes (voltage == NaN -> voltage != voltage),
        //   set it to threshold before it corrupts the recovery variable
        voltage = (voltage != voltage) ? IZ_SPIKE_THRESH : voltage;

        // Update recovery variable
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
    postsyn_traces[nid] = (spike) ? 1.0 : (postsyn_traces[nid] * TRACE_TAU);
    voltages[nid] = (spike) ? params[nid].c : voltage;
    recoveries[nid] = recovery + ((spike) ? params[nid].d : 0.0);
)

/******************************************************************************/
/************************* TRACE ACTIVATOR KERNELS ****************************/
/******************************************************************************/

#define AMPA_TAU        0.8    // tau = 5
#define GABAA_TAU       0.833  // tau = 6
#define NMDA_TAU        0.993  // tau = 150
#define GABAB_TAU       0.993  // tau = 150
#define MULT_TAU        0.95   // tau = 20
#define PLASTIC_TAU     0.933  // tau = 15

#define U_EXC 0.5
#define F_EXC 0.001       // 1 / 1000
#define D_EXC 0.00125     // 1 / 800

#define U_INH 0.2
#define F_INH 0.05        // 1 / 20
#define D_INH 0.00142857  // 1 / 700

// Extraction at start of kernel
#define ACTIV_EXTRACTIONS(STP_U, STP_F, STD_D) \
    IzhikevichAttributes *att = \
        (IzhikevichAttributes*)synapse_data.to_attributes; \
    float baseline_conductance = \
        att->baseline_conductance.get()[synapse_data.connection_index]; \
    float *delays = weights + (7*num_weights); \
\
    float *stds = weights + (4*num_weights); \
    float *stps = weights + (5*num_weights); \
\
    float stp_u = STP_U; \
    float stp_f = STP_F; \
    float std_d = STD_D;

#define ACTIV_EXTRACTIONS_SHORT(SHORT_NAME, SHORT_TAU) \
    float *short_conductances = att->SHORT_NAME.get(synapse_data.to_start_index); \
    float short_tau = SHORT_TAU; \
    float *short_traces = weights + (1*num_weights);

#define ACTIV_EXTRACTIONS_LONG(LONG_NAME, LONG_TAU) \
    float *long_conductances = att->LONG_NAME.get(synapse_data.to_start_index); \
    float long_tau = LONG_TAU; \
    float *long_traces  = weights + (2*num_weights); \

// Neuron Pre Operation
#define INIT_SUM \
    float short_sum = 0.0; \
    float long_sum = 0.0;

// Weight Operation
#define CALC_VAL_PREAMBLE \
    bool spike = extractor(outputs[from_index], delays[weight_index]) > 0.0; \
\
    float std    = stds[weight_index]; \
    float stp    = stps[weight_index]; \
    float weight = weights[weight_index] * stp * std;

#define CALC_VAL_SHORT \
    float short_trace = short_traces[weight_index] \
        + (spike ? (weight * baseline_conductance) : 0); \
    short_sum += short_trace; \
    short_traces[weight_index] = short_trace * short_tau;

#define CALC_VAL_LONG \
    float long_trace = long_traces[weight_index] \
        + (spike ? (weight * baseline_conductance) : 0); \
    long_sum += long_trace; \
    long_traces[weight_index] = long_trace * long_tau;

#define CALC_VAL_PLASTIC \
    stds[weight_index] += \
        ((1 - std) * std_d) \
        - ((spike) ? (std * stp) : 0); \
    stps[weight_index] += \
        ((stp_u - stp) * stp_f) \
        + ((spike) ? (stp_u * (1 - stp)) : 0);

// Neuron Post Operation
#define AGGREGATE_SHORT \
    short_conductances[to_index] += short_sum;

#define AGGREGATE_LONG \
    long_conductances[to_index] += long_sum;


/* Trace versions of activator functions */
CALC_ALL(activate_iz_add,
    ACTIV_EXTRACTIONS(U_EXC, F_EXC, D_EXC)
    ACTIV_EXTRACTIONS_SHORT(
        ampa_conductance,
        AMPA_TAU)
    ACTIV_EXTRACTIONS_LONG(
        nmda_conductance,
        NMDA_TAU),

    INIT_SUM,

    CALC_VAL_PREAMBLE
    CALC_VAL_SHORT
    CALC_VAL_LONG
    CALC_VAL_PLASTIC,

    AGGREGATE_SHORT
    AGGREGATE_LONG
);

CALC_ALL(activate_iz_sub,
    ACTIV_EXTRACTIONS(U_INH, F_INH, D_INH)
    ACTIV_EXTRACTIONS_SHORT(
        gabaa_conductance,
        GABAA_TAU)
    ACTIV_EXTRACTIONS_LONG(
        gabab_conductance,
        GABAB_TAU),

    INIT_SUM,

    CALC_VAL_PREAMBLE
    CALC_VAL_SHORT
    CALC_VAL_LONG
    CALC_VAL_PLASTIC,

    AGGREGATE_SHORT
    AGGREGATE_LONG
);

CALC_ALL(activate_iz_mult,
    ACTIV_EXTRACTIONS(U_EXC, F_EXC, D_EXC)
    ACTIV_EXTRACTIONS_SHORT(
        multiplicative_factor,
        MULT_TAU),

    INIT_SUM,

    CALC_VAL_PREAMBLE
    CALC_VAL_SHORT
    CALC_VAL_PLASTIC,

    AGGREGATE_SHORT
);

Kernel<SYNAPSE_ARGS> IzhikevichAttributes::get_activator(
        Connection *conn, DendriticNode *node) {
    if (node->is_second_order())
        ErrorManager::get_instance()->log_error(
            "Unimplemented connection type!");

    std::map<ConnectionType, std::map<Opcode, Kernel<SYNAPSE_ARGS> > > funcs;
    funcs[FULLY_CONNECTED][ADD]  = get_activate_iz_add_fully_connected();
    funcs[FULLY_CONNECTED][SUB]  = get_activate_iz_sub_fully_connected();
    funcs[FULLY_CONNECTED][MULT] = get_activate_iz_mult_fully_connected();
    funcs[ONE_TO_ONE][ADD]       = get_activate_iz_add_one_to_one();
    funcs[ONE_TO_ONE][SUB]       = get_activate_iz_sub_one_to_one();
    funcs[ONE_TO_ONE][MULT]      = get_activate_iz_mult_one_to_one();
    funcs[CONVERGENT][ADD]       = get_activate_iz_add_convergent();
    funcs[CONVERGENT][SUB]       = get_activate_iz_sub_convergent();
    funcs[CONVERGENT][MULT]      = get_activate_iz_mult_convergent();
    funcs[DIVERGENT][ADD]        = get_activate_iz_add_divergent();
    funcs[DIVERGENT][SUB]        = get_activate_iz_sub_divergent();
    funcs[DIVERGENT][MULT]       = get_activate_iz_mult_divergent();

    try {
        return funcs.at(conn->type).at(conn->opcode);
    } catch (...) {
        ErrorManager::get_instance()->log_error(
            "Unimplemented connection type!");
    }
}

/******************************************************************************/
/************************** TRACE UPDATER KERNELS *****************************/
/******************************************************************************/

#define UPDATE_EXTRACTIONS \
    float *presyn_traces = weights + (3*num_weights); \
    float *deltas        = weights + (6*num_weights); \
\
    IzhikevichAttributes *att = \
        (IzhikevichAttributes*)synapse_data.to_attributes; \
    float *to_traces = att->postsyn_trace.get(synapse_data.to_start_index); \
    float learning_rate = \
        att->learning_rate.get()[synapse_data.connection_index]; \

#define GET_DEST_ACTIVITY \
    float dest_trace = to_traces[to_index]; \
    float to_power = 0.0; \
    bool dest_spike = extractor(destination_outputs[to_index], 0) > 0.0;

#define UPDATE_WEIGHT \
    float weight = weights[weight_index]; \
    if (weight >= 0.0001) { \
        bool src_spike = extractor(outputs[from_index], 0); \
        float delta  = deltas[weight_index]; \
    \
        float src_trace = (src_spike) ? 1.0 : presyn_traces[weight_index]; \
        presyn_traces[weight_index] = src_trace * PLASTIC_TAU; \
    \
        delta += (dest_spike) ? (src_trace  * learning_rate) : 0.0; \
        delta -= (src_spike)  ? (dest_trace * learning_rate) : 0.0; \
        deltas[weight_index] -= (delta - 0.000001) * 0.0001; \
        weight += delta; \
        weights[weight_index] = \
            (weight < 0.0001) ? 0.0 \
                : (weight > max_weight) ? max_weight : weight; \
    }

CALC_ALL(update_iz_add,
    UPDATE_EXTRACTIONS,
    GET_DEST_ACTIVITY,
    UPDATE_WEIGHT,
; );

Kernel<SYNAPSE_ARGS> IzhikevichAttributes::get_updater(
        Connection *conn, DendriticNode *node) {
    if (node->is_second_order())
        ErrorManager::get_instance()->log_error(
            "Unimplemented connection type!");

    std::map<ConnectionType, std::map<Opcode, Kernel<SYNAPSE_ARGS> > > funcs;
    funcs[FULLY_CONNECTED][ADD]  = get_update_iz_add_fully_connected();
    funcs[ONE_TO_ONE][ADD]       = get_update_iz_add_one_to_one();
    funcs[CONVERGENT][ADD]       = get_update_iz_add_convergent();
    funcs[DIVERGENT][ADD]        = get_update_iz_add_divergent();

    try {
        return funcs.at(conn->type).at(conn->opcode);
    } catch (...) {
        ErrorManager::get_instance()->log_error(
            "Unimplemented connection type!");
    }
}

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

static std::string extract_parameter(Layer *layer,
        std::string key, std::string default_val) {
    try {
        return layer->get_config()->get_property(key);
    } catch (...) {
        ErrorManager::get_instance()->log_warning(
            "Unspecified parameter: " + key + " for layer \""
            + layer->name + "\" -- using " + default_val + ".");
        return default_val;
    }
}

static std::string extract_parameter(Connection *conn,
        std::string key, std::string default_val) {
    try {
        return conn->get_config()->get_property(key);
    } catch (...) {
        ErrorManager::get_instance()->log_warning(
            "Unspecified parameter: " + key + " for conn \""
            + conn->from_layer->name + "\" -> \""
            + conn->to_layer->name + "\" -- using "
            + default_val + ".");
        return default_val;
    }
}

static void check_parameters(Layer *layer) {
    std::set<std::string> valid_params;
    valid_params.insert("init");
    valid_params.insert("spacing");

    for (auto pair : layer->get_config()->get_properties())
        if (valid_params.count(pair.first) == 0)
            ErrorManager::get_instance()->log_error(
                "Unrecognized layer parameter: " + pair.first);
}

static void check_parameters(Connection *conn) {
    std::set<std::string> valid_params;
    valid_params.insert("conductance");
    valid_params.insert("learning rate");
    valid_params.insert("myelinated");
    valid_params.insert("x offset");
    valid_params.insert("y offset");

    for (auto pair : conn->get_config()->get_properties())
        if (valid_params.count(pair.first) == 0)
            ErrorManager::get_instance()->log_error(
                "Unrecognized connection parameter: " + pair.first);
}

IzhikevichAttributes::IzhikevichAttributes(LayerList &layers)
        : Attributes(layers, BIT) {
    // Count connections
    int num_connections = 0;
    for (auto& layer : layers)
        num_connections += layer->get_input_connections().size();

    // Baseline conductances
    this->baseline_conductance = Pointer<float>(num_connections);
    Attributes::register_variable(&this->baseline_conductance);

    // Learning rate
    this->learning_rate = Pointer<float>(num_connections);
    Attributes::register_variable(&this->learning_rate);

    // Conductances
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
    this->postsyn_trace = Pointer<float>(total_neurons);
    Attributes::register_variable(&this->recovery);
    Attributes::register_variable(&this->voltage);
    Attributes::register_variable(&this->postsyn_trace);

    // Neuron parameters
    this->neuron_parameters = Pointer<IzhikevichParameters>(total_neurons);
    Attributes::register_variable(&this->neuron_parameters);

    // Fill in table
    int start_index = 0;
    for (auto& layer : layers) {
        // Check layer parameters
        check_parameters(layer);

        IzhikevichParameters params = create_parameters(
            extract_parameter(layer, "init", "regular"));
        for (int j = 0 ; j < layer->size ; ++j) {
            neuron_parameters[start_index+j] = params;
            postsyn_trace[start_index+j] = 0.0;

            // Run simulation to stable point
            float v = params.c;
            float r = params.b * params.c;
            float delta_v = 1.0;
            float delta_r = 1.0;
            do {
                delta_v = (0.04 * v * v) + (5*v) + 140 - r;
                v += delta_v;

                delta_r = params.a * ((params.b * v) - r);
                r += delta_r;
            } while (abs(delta_v) > 0.001 and abs(delta_r) > 0.001);

            voltage[start_index+j] = v;
            recovery[start_index+j] = r;
        }
        start_index += layer->size;

        // Connection properties
        for (auto& conn : layer->get_input_connections()) {
            // Check connection parameters
            check_parameters(conn);

            // Retrieve baseline conductance
            baseline_conductance[connection_indices[conn->id]] =
                std::stof(extract_parameter(conn, "conductance", "1.0"));

            // Retrieve learning rate
            learning_rate[connection_indices[conn->id]] =
                std::stof(extract_parameter(conn, "learning rate", "0.004"));
        }
    }
}

void IzhikevichAttributes::set_delays(Connection *conn, float* delays) {
    int base_delay = conn->delay;

    // Myelinated connections use the base delay only
    if (extract_parameter(conn, "myelinated", "") != "") {
        for (int i = 0 ; i < conn->get_num_weights() ; ++i)
            delays[i] = base_delay;
        return;
    }

    float velocity = 0.15;
    float from_spacing = std::stof(
        extract_parameter(conn->from_layer, "spacing", "0.09"));
    float to_spacing = std::stof(
        extract_parameter(conn->to_layer, "spacing", "0.09"));
    float x_offset = std::stof(
        extract_parameter(conn, "x offset", "0.0"));
    float y_offset = std::stof(
        extract_parameter(conn, "x offset", "0.0"));

    switch(conn->type) {
        case(FULLY_CONNECTED): {
            auto fcc = conn->get_config()->get_fully_connected_config();
            int from_columns = conn->from_layer->columns;
            int weight_index = 0;
            for (int f_row = fcc->from_row_start; f_row < fcc->from_row_end; ++f_row) {
                float f_y = f_row * from_spacing;

                for (int f_col = fcc->from_col_start; f_col < fcc->from_col_end; ++f_col) {
                    float f_x = f_col * from_spacing;

                    for (int t_row = fcc->to_row_start; t_row < fcc->to_row_end; ++t_row) {
                        float t_y = t_row * to_spacing + y_offset;

                        for (int t_col = fcc->to_col_start; t_col < fcc->to_col_end; ++t_col) {
                            float t_x = t_col * to_spacing + x_offset;

                            float distance = pow(
                                pow(t_x - f_x, 2) + pow(t_y - f_y, 2),
                                0.5);
                            int delay = base_delay + (distance / velocity);
                            if (delay > 31)
                                ErrorManager::get_instance()->log_error(
                                    "Unmyelinated axons cannot have delays "
                                    "greater than 31!");
                            delays[weight_index] = delay;
                            ++weight_index;
                        }
                    }
                }
            }
            break;
        }
        case(ONE_TO_ONE):
            for (int i = 0 ; i < conn->get_num_weights() ; ++i)
                delays[i] = base_delay;
            break;
        case(CONVERGENT): {
            auto ac = conn->get_config()->get_arborized_config();
            int to_size = conn->to_layer->size;
            int field_size = ac->row_field_size * ac->column_field_size;

            if (ac->row_stride != ac->column_stride
                or (int(to_spacing / from_spacing) != ac->row_stride))
                ErrorManager::get_instance()->log_error(
                    "Spacing and strides must match up for convergent connection!");

            for (int f_row = 0; f_row < ac->row_field_size; ++f_row) {
                float f_y = (f_row + ac->row_offset) * to_spacing;

                for (int f_col = 0; f_col < ac->column_field_size; ++f_col) {
                    float f_x = (f_col + ac->column_offset) * to_spacing;

                    float distance = pow(
                        pow(f_x, 2) + pow(f_y, 2),
                        0.5);
                    int delay = base_delay + (distance / velocity);
                    if (delay > 31)
                        ErrorManager::get_instance()->log_error(
                            "Unmyelinated axons cannot have delays "
                            "greater than 31!");

                    int f_index = (f_row * ac->column_field_size) + f_col;

                    for (int i = 0 ; i < to_size ; ++i) {
#ifdef __CUDACC__
                        delays[(f_index * to_size) + i] = delay;
#else
                        delays[(i * field_size) + f_index] = delay;
#endif
                    }
                }
            }
            break;
        }
        case(DIVERGENT): {
            auto ac = conn->get_config()->get_arborized_config();
            int num_weights = conn->get_num_weights();
            int to_rows = conn->to_layer->rows;
            int to_columns = conn->to_layer->columns;
            int to_size = conn->to_layer->size;
            int from_rows = conn->from_layer->rows;
            int from_columns = conn->from_layer->columns;

            if (ac->row_stride != ac->column_stride
                or (int(from_spacing / to_spacing) != ac->row_stride))
                ErrorManager::get_instance()->log_error(
                    "Spacing and strides must match up for divergent connection!");


            int row_field_size = ac->row_field_size;
            int column_field_size = ac->column_field_size;
            int row_stride = ac->row_stride;
            int column_stride = ac->column_stride;
            int row_offset = ac->row_offset;
            int column_offset = ac->column_offset;
            int kernel_size = row_field_size * column_field_size;

            for (int d_row = 0 ; d_row < to_rows ; ++d_row) {
                for (int d_col = 0 ; d_col < to_columns ; ++d_col) {
                    int to_index = d_row*to_columns + d_col;
                    /* Determine range of source neurons for divergent kernel */
                    int start_s_row = (d_row - row_offset - row_field_size + row_stride) / row_stride;
                    int start_s_col = (d_col - column_offset - column_field_size + column_stride) / column_stride;
                    int end_s_row = (d_row - row_offset) / row_stride;
                    int end_s_col = (d_col - column_offset) / column_stride;

                    // SERIAL
                    int weight_offset = to_index * num_weights / to_size;
                    // PARALLEL
                    int kernel_row_size = num_weights / to_size;

                    /* Iterate over relevant source neurons... */
                    int k_index = 0;
                    for (int s_row = start_s_row ; s_row <= end_s_row ; ++s_row) {
                        for (int s_col = start_s_col ; s_col <= end_s_col ; (++s_col, ++k_index)) {
                            /* Avoid making connections with non-existent neurons! */
                            if (s_row < 0 or s_row >= from_rows
                                or s_col < 0 or s_col >= from_columns)
                                continue;

                            int from_index = (s_row * from_columns) + s_col;

                            float d_x = abs(
                                ((d_row + ac->row_offset) * to_spacing)
                                - (s_row * from_spacing));
                            float d_y = abs(
                                ((d_col + ac->column_offset) * to_spacing)
                                - (s_col * from_spacing));

                            float distance = pow(
                                pow(d_x, 2) + pow(d_y, 2),
                                0.5);
                            int delay = base_delay + (distance / velocity);
                            if (delay > 31)
                                ErrorManager::get_instance()->log_error(
                                    "Unmyelinated axons cannot have delays "
                                    "greater than 31!");
#ifdef __CUDACC__
                            int weight_index = to_index + (k_index * kernel_row_size);
#else
                            int weight_index = weight_offset + k_index;
#endif
                            delays[weight_index] = delay;
                        }
                    }
                }
            }
            break;
        }
    }
}

void IzhikevichAttributes::process_weight_matrix(WeightMatrix* matrix) {
    Connection *conn = matrix->connection;
    Pointer<float> mData = matrix->get_data();

    int num_weights = conn->get_num_weights();

    // Short term trace
    clear_weights(mData + 1*num_weights, num_weights);

    // Long term trace
    clear_weights(mData + 2*num_weights, num_weights);

    // Plasticity trace
    clear_weights(mData + 3*num_weights, num_weights);

    // Short Term Depression
    set_weights(mData + 4*num_weights, num_weights, 1.0);

    // Short Term Potentiation
    if (conn->opcode == ADD)
        set_weights(mData + 5*num_weights, num_weights, U_EXC);
    else if (conn->opcode == SUB)
        set_weights(mData + 5*num_weights, num_weights, U_INH);

    // Weight Delta
    set_weights(mData + 6*num_weights, num_weights, 0.000001);

    // Delays
    set_delays(conn, mData + 7*num_weights);
}
