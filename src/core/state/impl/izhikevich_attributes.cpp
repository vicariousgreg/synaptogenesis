#include <string>
#include <math.h>

#include "state/impl/izhikevich_attributes.h"
#include "state/weight_matrix.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/tools.h"

REGISTER_ATTRIBUTES(IzhikevichAttributes, "izhikevich", BIT)
REGISTER_WEIGHT_MATRIX(IzhikevichWeightMatrix, "izhikevich")

/******************************************************************************/
/******************************** PARAMS **************************************/
/******************************************************************************/

/* Neuron parameters class.
 * Contains a,b,c,d parameters for Izhikevich model */
class IzhikevichParameters {
    public:
        IzhikevichParameters(float a, float b, float c, float d) :
                a(a), b(b), c(c), d(d) {}
        float a, b, c, d;
};

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

static void create_parameters(std::string str,
        float* as, float* bs, float* cs, float* ds, int size) {
    if (str == "random positive") {
        // (ai; bi) = (0:02; 0:2) and (ci; di) = (-65; 8) + (15;-6)r2
        for (int i = 0 ; i < size ; ++i) as[i] = 0.02;

        // increase for higher frequency oscillations
        for (int i = 0 ; i < size ; ++i) bs[i] = 0.2;

        for (int i = 0 ; i < size ; ++i)
            cs[i] = -65.0 + (15.0 * pow(fRand(), 2));

        for (int i = 0 ; i < size ; ++i)
            ds[i] = 8.0 - (6.0 * pow(fRand(), 2));
    } else if (str == "random negative") {
        //(ai; bi) = (0:02; 0:25) + (0:08;-0:05)ri and (ci; di)=(-65; 2).
        for (int i = 0 ; i < size ; ++i)
            as[i] = 0.02 + (0.08 * fRand());

        for (int i = 0 ; i < size ; ++i)
            bs[i] = 0.25 - (0.05 * fRand());

        for (int i = 0 ; i < size ; ++i) cs[i] = -65.0;
        for (int i = 0 ; i < size ; ++i) ds[i] = 2.0;
    } else {
        IzhikevichParameters params(0,0,0,0);
        if (str == "default")                 params = DEFAULT;
        else if (str == "regular")            params = REGULAR;
        else if (str == "bursting")           params = BURSTING;
        else if (str == "chattering")         params = CHATTERING;
        else if (str == "fast")               params = FAST;
        else if (str == "low_threshold")      params = LOW_THRESHOLD;
        else if (str == "thalamo_cortical")   params = THALAMO_CORTICAL;
        else if (str == "resonator")          params = RESONATOR;
        else
            LOG_ERROR(
                "Unrecognized parameter string: " + str);


        for (int i = 0 ; i < size ; ++i) as[i] = params.a;
        for (int i = 0 ; i < size ; ++i) bs[i] = params.b;
        for (int i = 0 ; i < size ; ++i) cs[i] = params.c;
        for (int i = 0 ; i < size ; ++i) ds[i] = params.d;
    }
}

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

/* Voltage threshold for neuron spiking. */
#define IZ_SPIKE_THRESH 30

/* Euler resolution for voltage update. */
#define IZ_EULER_RES 10
#define IZ_EULER_RES_INV 0.1

/* Time dynamics of postsynaptic spikes */
#define TRACE_TAU 0.933  // 15

/* Time dynamics of dopamine */
#define DOPAMINE_CLEAR_TAU 0.95  // 20

/* Time dynamics of acetylcholine */
#define ACETYLCHOLINE_CLEAR_TAU 0.95  // 20

/* STDP A constant */
#define STDP_A 0.004

BUILD_ATTRIBUTE_KERNEL(IzhikevichAttributes, iz_attribute_kernel,
    IzhikevichAttributes *iz_att = (IzhikevichAttributes*)att;

    float *ampa_conductances = iz_att->ampa_conductance.get();
    float *nmda_conductances = iz_att->nmda_conductance.get();
    float *gabaa_conductances = iz_att->gabaa_conductance.get();
    float *gabab_conductances = iz_att->gabab_conductance.get();
    float *multiplicative_factors = iz_att->multiplicative_factor.get();
    float *dopamines = iz_att->dopamine.get();
    float *acetylcholines = iz_att->acetylcholine.get();

    float *voltages = iz_att->voltage.get();
    float *recoveries = iz_att->recovery.get();
    float *postsyn_traces = iz_att->postsyn_trace.get();
    unsigned int *spikes = (unsigned int*)outputs;
    float *as = iz_att->as.get();
    float *bs = iz_att->bs.get();
    float *cs = iz_att->cs.get();
    float *ds = iz_att->ds.get();

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

    float a = as[nid];
    float b = bs[nid];

    // Euler's method for voltage/recovery update
    // If the voltage exceeds the spiking threshold, break
    for (int i = 0 ; (i < IZ_EULER_RES) and (voltage < IZ_SPIKE_THRESH) ; ++i) {
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
        voltage += delta_v * IZ_EULER_RES_INV;

        // If the voltage explodes (voltage == NaN -> voltage != voltage),
        //   set it to threshold before it corrupts the recovery variable
        voltage = (voltage != voltage) ? IZ_SPIKE_THRESH : voltage;

        float adjusted_tau = (voltage > IZ_SPIKE_THRESH)
            ? delta_v / (IZ_SPIKE_THRESH - voltage + delta_v)
            : IZ_EULER_RES_INV;

        // Update recovery variable
        recovery += a * ((b * voltage) - recovery) * adjusted_tau;
    }

    ampa_conductances[nid] = 0.0;
    nmda_conductances[nid] = 0.0;
    gabaa_conductances[nid] = 0.0;
    gabab_conductances[nid] = 0.0;
    multiplicative_factors[nid] = 0.0;
    dopamines[nid] *= DOPAMINE_CLEAR_TAU;
    acetylcholines[nid] *= ACETYLCHOLINE_CLEAR_TAU;

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
    postsyn_traces[nid] = (spike) ? STDP_A : (postsyn_traces[nid] * TRACE_TAU);
    voltages[nid] = (spike) ? cs[nid] : voltage;
    recoveries[nid] = recovery + (spike * ds[nid]);
)

/******************************************************************************/
/************************* TRACE ACTIVATOR KERNELS ****************************/
/******************************************************************************/

#define AMPA_TAU          0.8    // tau = 5
#define GABAA_TAU         0.833  // tau = 6
#define NMDA_TAU          0.993  // tau = 150
#define GABAB_TAU         0.993  // tau = 150

#define MULT_TAU          0.95   // tau = 20
#define DOPAMINE_TAU      0.95   // tau = 20
#define ACETYLCHOLINE_TAU 0.95   // tau = 20
#define PLASTIC_TAU       0.95   // tau = 20

#define U_DEPRESS 0.5
#define F_DEPRESS 0.001       // 1 / 1000
#define D_DEPRESS 0.00125     // 1 / 800

#define U_POTENTIATE 0.2
#define F_POTENTIATE 0.05     // 1 / 20
#define D_POTENTIATE 0.001429 // 1 / 700

// Extraction at start of kernel
#define ACTIV_EXTRACTIONS \
    IzhikevichAttributes *att = \
        (IzhikevichAttributes*)synapse_data.attributes; \
    IzhikevichWeightMatrix *matrix = \
        (IzhikevichWeightMatrix*)synapse_data.matrix; \
    float baseline_conductance = matrix->baseline_conductance; \
    bool stp_flag = matrix->stp_flag; \
\
    float *stds   = matrix->stds.get(); \
    float *stps   = matrix->stps.get(); \
    int   *delays = matrix->delays.get();

#define ACTIV_EXTRACTIONS_SHORT(SHORT_NAME, SHORT_TAU) \
    float *short_conductances = att->SHORT_NAME.get(); \
    float short_tau = SHORT_TAU; \
    float *short_traces   = matrix->short_traces.get();

#define ACTIV_EXTRACTIONS_LONG(LONG_NAME, LONG_TAU) \
    float *long_conductances = att->LONG_NAME.get(); \
    float long_tau = LONG_TAU; \
    float *long_traces   = matrix->long_traces.get();

// Neuron Pre Operation
#define INIT_SUM \
    float short_sum = 0.0; \
    float long_sum = 0.0;

// Weight Operation
#define CALC_VAL_PREAMBLE \
    float spike = extractor(outputs[from_index], delays[weight_index]); \
\
    float std    = stds[weight_index]; \
    float stp    = stps[weight_index]; \
    float weight = weights[weight_index] * stp * std;

#define CALC_VAL_SHORT \
    float short_trace = short_traces[weight_index] \
        + (spike * weight * baseline_conductance); \
    short_sum += short_trace; \
    short_traces[weight_index] = short_trace * short_tau;

#define CALC_VAL_LONG \
    float long_trace = long_traces[weight_index] \
        + (spike * weight * baseline_conductance); \
    long_sum += long_trace; \
    long_traces[weight_index] = long_trace * long_tau;

#define CALC_VAL_PLASTIC(STP_U, STP_F, STD_D) \
    if (stp_flag) { \
        stds[weight_index] += \
            ((1 - std) * STD_D) \
            - (spike * std * stp); \
        stps[weight_index] += \
            ((STP_U - stp) * STP_F) \
            + (spike * STP_U * (1 - stp)); \
    }

// Neuron Post Operation
#define AGGREGATE_SHORT \
    short_conductances[to_index] += short_sum;

#define AGGREGATE_LONG \
    long_conductances[to_index] += long_sum;


/* Trace versions of activator functions */
CALC_ALL(activate_iz_add,
    ACTIV_EXTRACTIONS
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
    CALC_VAL_PLASTIC(U_DEPRESS, F_DEPRESS, D_DEPRESS),

    AGGREGATE_SHORT
    AGGREGATE_LONG
);

CALC_ALL(activate_iz_sub,
    ACTIV_EXTRACTIONS
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
    CALC_VAL_PLASTIC(U_POTENTIATE, F_POTENTIATE, D_POTENTIATE),

    AGGREGATE_SHORT
    AGGREGATE_LONG
);

CALC_ALL(activate_iz_mult,
    ACTIV_EXTRACTIONS
    ACTIV_EXTRACTIONS_SHORT(
        multiplicative_factor,
        MULT_TAU),

    INIT_SUM,

    CALC_VAL_PREAMBLE
    CALC_VAL_SHORT
    CALC_VAL_PLASTIC(U_DEPRESS, F_DEPRESS, D_DEPRESS),

    AGGREGATE_SHORT
);

CALC_ALL(activate_iz_reward,
    ACTIV_EXTRACTIONS
    ACTIV_EXTRACTIONS_SHORT(
        dopamine,
        DOPAMINE_TAU),

    INIT_SUM,

    CALC_VAL_PREAMBLE
    CALC_VAL_SHORT
    CALC_VAL_PLASTIC(U_DEPRESS, F_DEPRESS, D_DEPRESS),

    AGGREGATE_SHORT
);

CALC_ALL(activate_iz_modulate,
    ACTIV_EXTRACTIONS
    ACTIV_EXTRACTIONS_SHORT(
        acetylcholine,
        ACETYLCHOLINE_TAU),

    INIT_SUM,

    CALC_VAL_PREAMBLE
    CALC_VAL_SHORT
    CALC_VAL_PLASTIC(U_DEPRESS, F_DEPRESS, D_DEPRESS),

    AGGREGATE_SHORT
);

CALC_ALL(activate_iz_gap,
    IzhikevichAttributes *att =
        (IzhikevichAttributes*)synapse_data.attributes;
    float *voltage = att->voltage.get();,

    float v = voltage[to_index];
    float sum = 0.0;,

    float weight = weights[weight_index];
    sum += (voltage[from_index] - v) * weight;,

    inputs[to_index] += sum;
);

Kernel<SYNAPSE_ARGS> IzhikevichAttributes::get_activator(Connection *conn) {
    // These are not supported because of the change of weight matrix pointer
    // Second order host connections require their weight matrices to be copied
    // Currently, this only copies the first matrix in the stack, and this
    //   attribute set uses auxiliary data
    if (conn->second_order)
        LOG_ERROR(
            "Unimplemented connection type!");

    std::map<ConnectionType, std::map<Opcode, Kernel<SYNAPSE_ARGS>>> funcs;
    funcs[FULLY_CONNECTED][ADD]      = get_activate_iz_add_fully_connected();
    funcs[FULLY_CONNECTED][SUB]      = get_activate_iz_sub_fully_connected();
    funcs[FULLY_CONNECTED][MULT]     = get_activate_iz_mult_fully_connected();
    funcs[FULLY_CONNECTED][REWARD]   = get_activate_iz_reward_fully_connected();
    funcs[FULLY_CONNECTED][MODULATE] = get_activate_iz_modulate_fully_connected();
    funcs[SUBSET][ADD]               = get_activate_iz_add_subset();
    funcs[SUBSET][SUB]               = get_activate_iz_sub_subset();
    funcs[SUBSET][MULT]              = get_activate_iz_mult_subset();
    funcs[ONE_TO_ONE][ADD]           = get_activate_iz_add_one_to_one();
    funcs[ONE_TO_ONE][SUB]           = get_activate_iz_sub_one_to_one();
    funcs[ONE_TO_ONE][MULT]          = get_activate_iz_mult_one_to_one();
    funcs[CONVERGENT][ADD]           = get_activate_iz_add_convergent();
    funcs[CONVERGENT][SUB]           = get_activate_iz_sub_convergent();
    funcs[CONVERGENT][MULT]          = get_activate_iz_mult_convergent();
    funcs[CONVERGENT][GAP]           = get_activate_iz_gap_convergent();
    funcs[DIVERGENT][ADD]            = get_activate_iz_add_divergent();
    funcs[DIVERGENT][SUB]            = get_activate_iz_sub_divergent();
    funcs[DIVERGENT][MULT]           = get_activate_iz_mult_divergent();

    try {
        return funcs.at(conn->type).at(conn->opcode);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Unimplemented connection type!");
    }
}

/******************************************************************************/
/************************** TRACE UPDATER KERNELS *****************************/
/******************************************************************************/

#define UPDATE_EXTRACTIONS \
    IzhikevichWeightMatrix *matrix = \
        (IzhikevichWeightMatrix*)synapse_data.matrix; \
    float *presyn_traces   = matrix->presyn_traces.get(); \
    float *eligibilities   = matrix->eligibilities.get(); \
    int   *delays = matrix->delays.get(); \
\
    IzhikevichAttributes *att = \
        (IzhikevichAttributes*)synapse_data.attributes; \
    float *to_traces = att->postsyn_trace.get(); \
    float *dopamines = att->dopamine.get(); \
    float *acetylcholines = att->acetylcholine.get(); \
    float learning_rate = matrix->learning_rate; \

#define GET_DEST_ACTIVITY \
    float dest_trace = to_traces[to_index]; \
    float dopamine = dopamines[to_index]; \
    float acetylcholine = acetylcholines[to_index]; \
    float dest_spike = extractor(destination_outputs[to_index], 0);

/* Minimum weight */
#define MIN_WEIGHT 0.0001

/* Time dynamics for long term eligibility trace */
#define C_TAU 0.99

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
        float weight_delta = \
                (dest_spike * src_trace) \
                - (src_spike  * dest_trace); \
\
        /* Update eligibility trace */ \
        float c = \
            eligibilities[weight_index] = \
                (eligibilities[weight_index] * C_TAU) \
                + weight_delta; \
\
        /* Add dopamine driven long term changes */ \
        weight_delta += c * dopamine; \
\
        /* Add acetylcholine driven short term changes */ \
        weight_delta += c * acetylcholine; \
\
        /* Calculate new weight */ \
        weight += learning_rate * weight_delta; \
\
        /* Ensure weight stays within boundaries */ \
        weights[weight_index] = weight = \
            MAX(MIN_WEIGHT, MIN(max_weight, weight)); \
    }

CALC_ALL(update_iz_add,
    UPDATE_EXTRACTIONS,
    GET_DEST_ACTIVITY,
    UPDATE_WEIGHT,
; );

Kernel<SYNAPSE_ARGS> IzhikevichAttributes::get_updater(Connection *conn) {
    // Second order and convolutional updaters are not currently supported
    if (conn->second_order or conn->convolutional)
        LOG_ERROR(
            "Unimplemented connection type!");

    std::map<ConnectionType, std::map<Opcode, Kernel<SYNAPSE_ARGS>>> funcs;
    funcs[FULLY_CONNECTED][ADD]  = get_update_iz_add_fully_connected();
    funcs[SUBSET][ADD]           = get_update_iz_add_subset();
    funcs[ONE_TO_ONE][ADD]       = get_update_iz_add_one_to_one();
    funcs[CONVERGENT][ADD]       = get_update_iz_add_convergent();
    funcs[DIVERGENT][ADD]        = get_update_iz_add_divergent();

    try {
        return funcs.at(conn->type).at(conn->opcode);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Unimplemented connection type!");
    }
}

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

static void check_parameters(Layer *layer) {
    std::set<std::string> valid_params;
    valid_params.insert("name");
    valid_params.insert("neural model");
    valid_params.insert("rows");
    valid_params.insert("columns");

    valid_params.insert("init");
    valid_params.insert("neuron spacing");

    for (auto pair : layer->get_config()->get())
        if (valid_params.count(pair.first) == 0)
            LOG_WARNING(
                "Unrecognized layer parameter: " + pair.first);
}

static void check_parameters(Connection *conn) {
    std::set<std::string> valid_params;
    valid_params.insert("from layer");
    valid_params.insert("from structure");
    valid_params.insert("to layer");
    valid_params.insert("to structure");
    valid_params.insert("max");
    valid_params.insert("plastic");
    valid_params.insert("opcode");
    valid_params.insert("type");

    valid_params.insert("conductance");
    valid_params.insert("learning rate");
    valid_params.insert("myelinated");
    valid_params.insert("random delay");
    valid_params.insert("cap delay");
    valid_params.insert("x offset");
    valid_params.insert("y offset");
    valid_params.insert("short term plasticity");

    for (auto pair : conn->get_config()->get())
        if (valid_params.count(pair.first) == 0)
            LOG_WARNING(
                "Unrecognized connection parameter: " + pair.first);
}

IzhikevichAttributes::IzhikevichAttributes(Layer *layer)
        : Attributes(layer, BIT) {
    // Conductances
    this->ampa_conductance = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("ampa", &ampa_conductance);
    this->nmda_conductance = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("nmda", &nmda_conductance);
    this->gabaa_conductance = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("gabaa", &gabaa_conductance);
    this->gabab_conductance = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("gabab", &gabab_conductance);
    this->multiplicative_factor = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("mult", &multiplicative_factor);
    this->dopamine = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("dopamine", &dopamine);
    this->acetylcholine = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("acetylcholine", &acetylcholine);

    // Neuron variables
    this->voltage = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("voltage", &voltage);
    this->recovery = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("recovery", &recovery);
    this->postsyn_trace = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("post trace", &postsyn_trace);

    // Neuron parameters
    this->as = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("a", &as);
    this->bs = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("b", &bs);
    this->cs = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("c", &cs);
    this->ds = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("d", &ds);

    // Check layer parameters
    check_parameters(layer);

    create_parameters(layer->get_parameter("init", "regular"),
        this->as, this->bs, this->cs, this->ds, layer->size);
    for (int j = 0 ; j < layer->size ; ++j) {
        postsyn_trace[j] = 0.0;

        // Run simulation to stable point
        float v = this->cs[j];
        float r = this->bs[j] * this->cs[j];
        float delta_v;
        float delta_r;
        float a = this->as[j];
        float b = this->bs[j];
        do {
            delta_v = (0.04 * v * v) + (5*v) + 140 - r;
            v += delta_v;

            delta_r = a * ((b * v) - r);
            r += delta_r;
        } while (abs(delta_v) > 0.001 and abs(delta_r) > 0.001);

        voltage[j] = v;
        recovery[j] = r;
    }

    // Connection properties
    for (auto& conn : layer->get_input_connections()) {
        // Check connection parameters
        check_parameters(conn);

        // Ensure gap junctions are self-connections
        if (conn->type == GAP and conn->from_layer != conn->to_layer)
            LOG_ERROR(
                "Error " + conn->str() + "\n"
                "Gap junctions must be between neurons of the same layer.");
    }
}

void IzhikevichWeightMatrix::register_variables() {
    // Short term (AMPA/GABAA) conductance trace
    this->short_traces = WeightMatrix::create_variable<float>();
    WeightMatrix::register_variable("short trace", &short_traces);

    // Long term (NMDA/GABAA) conductance trace
    this->long_traces = WeightMatrix::create_variable<float>();
    WeightMatrix::register_variable("long trace", &long_traces);

    // Presynaptic trace
    this->presyn_traces = WeightMatrix::create_variable<float>();
    WeightMatrix::register_variable("presyn trace", &presyn_traces);

    // Short Term Depression
    this->stds = WeightMatrix::create_variable<float>();
    WeightMatrix::register_variable("short term depression", &stds);

    // Short Term Potentiation
    this->stps = WeightMatrix::create_variable<float>();
    WeightMatrix::register_variable("short term potentiation", &stps);

    // Long term eligibility trace
    this->eligibilities = WeightMatrix::create_variable<float>();
    WeightMatrix::register_variable("eligibilities", &eligibilities);

    // Delay
    this->delays = WeightMatrix::create_variable<int>();
    WeightMatrix::register_variable("delays", &delays);
}

void IzhikevichAttributes::process_weight_matrix(WeightMatrix* matrix) {
    auto iz_mat = (IzhikevichWeightMatrix*)matrix;
    Connection *conn = matrix->connection;

    // Retrieve baseline conductance
    iz_mat->baseline_conductance =
        std::stof(conn->get_parameter("conductance", "1.0"));

    // Retrieve learning rate
    iz_mat->learning_rate =
        std::stof(conn->get_parameter("learning rate", "0.004"));

    // Retrieve short term plasticity flag
    iz_mat->stp_flag =
        conn->get_parameter("short term plasticity", "true") == "true";

    int num_weights = conn->get_num_weights();
    Pointer<float> mData = matrix->get_weights();

    // Short term trace
    clear_weights(iz_mat->short_traces, num_weights);

    // Long term trace
    clear_weights(iz_mat->long_traces, num_weights);

    // Plasticity trace
    clear_weights(iz_mat->presyn_traces, num_weights);

    // Short Term Depression
    set_weights(iz_mat->stds, num_weights, 1.0);

    // Short Term Potentiation
    if (iz_mat->stp_flag)
        set_weights(iz_mat->stps, num_weights, 1.0);
    else
        switch(conn->opcode) {
            case ADD:
            case MULT:
            case REWARD:
            case MODULATE:
                set_weights(iz_mat->stps, num_weights, U_DEPRESS);
                break;
            case SUB:
                set_weights(iz_mat->stps, num_weights, U_POTENTIATE);
                break;
        }

    // Long term eligibiity trace
    set_weights(iz_mat->eligibilities, num_weights, 0.0);

    // Delays
    // Myelinated connections use the base delay only
    int *delays = iz_mat->delays.get();
    if (conn->get_parameter("myelinated", "false") == "true") {
        int delay = conn->delay;
        for (int i = 0 ; i < num_weights ; ++i)
            delays[i] = delay;
    } else if (conn->get_parameter("random delay", "false") == "true") {
        int max_delay = std::stoi(
            conn->get_parameter("random delay", "0"));
        if (max_delay > 31)
            LOG_ERROR(
                "Randomized axons cannot have delays greater than 31!");
        iRand(delays, num_weights, 0, max_delay);
    } else {
        set_delays(BIT, conn, delays, 0.15,
            conn->get_parameter("cap delay", "false") == "true",
            std::stof(conn->from_layer->get_parameter("neuron spacing", "0.1")),
            std::stof(conn->to_layer->get_parameter("neuron spacing", "0.1")),
            std::stof(conn->get_parameter("x offset", "0.0")),
            std::stof(conn->get_parameter("y offset", "0.0")));
    }
}
