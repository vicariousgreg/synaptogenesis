#include <string>
#include <math.h>

#include "state/impl/spnet_attributes.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/tools.h"

REGISTER_ATTRIBUTES(SpnetAttributes, "spnet", BIT)
REGISTER_WEIGHT_MATRIX(SpnetWeightMatrix, "spnet")

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
#define IZ_SPIKE_THRESH 30.0

/* Euler resolution for voltage update. */
#define IZ_EULER_RES 10
#define IZ_EULER_RES_INV 0.1

/* Time dynamics of spike traces */
#define STDP_TAU_POS       0.95   // tau = 20ms
#define STDP_TAU_NEG       0.95   // tau = 20ms

/* STDP A constants */
#define STDP_A_POS 0.1
#define STDP_A_NEG 0.1

BUILD_ATTRIBUTE_KERNEL(SpnetAttributes, spnet_attribute_kernel,
    float *voltages = att->voltage.get();
    float *recoveries = att->recovery.get();
    float *postsyn_exc_traces = att->postsyn_exc_trace.get();
    int *time_since_spikes = att->time_since_spike.get();
    unsigned int *spikes = (unsigned int*)outputs;
    float *as = att->as.get();
    float *bs = att->bs.get();
    float *cs = att->cs.get();
    float *ds = att->ds.get();

    ,

    /**********************
     *** VOLTAGE UPDATE ***
     **********************/
    float voltage = voltages[nid];
    float recovery = recoveries[nid];
    float current = inputs[nid];

    float a = as[nid];
    float b = bs[nid];

    // Euler's method for voltage/recovery update
    // If the voltage exceeds the spiking threshold, break
    for (int i = 0 ; (i < IZ_EULER_RES) and (voltage < IZ_SPIKE_THRESH) ; ++i) {
        // Update voltage
        // For numerical stability, use hybrid numerical method
        //   (see section 5b of "Hybrid Spiking Models" by Izhikevich)
        float delta_v = (0.04 * voltage * voltage) +
                        (5*voltage) + 140 - recovery + current;

        voltage += delta_v * IZ_EULER_RES_INV;

        // If the voltage explodes (voltage == NaN -> voltage != voltage),
        //   set it to threshold before it corrupts the recovery variable
        voltage = (voltage != voltage) ? IZ_SPIKE_THRESH : voltage;

        float adjusted_tau = (voltage > IZ_SPIKE_THRESH)
            ? delta_v
                / (IZ_SPIKE_THRESH - voltage + delta_v)
                * IZ_EULER_RES_INV
            : IZ_EULER_RES_INV;

        // Update recovery variable
        recovery += a * adjusted_tau * ((b * voltage) - recovery);
    }

    /********************
     *** SPIKE UPDATE ***
     ********************/
    // Determine if spike occurred
    unsigned int spike = voltage >= IZ_SPIKE_THRESH;

    SHIFT_BIT_OUTPUTS(spikes, spike);

    // Update trace, voltage, recovery
    postsyn_exc_traces[nid] = (prev_bit)
        ? (postsyn_exc_traces[nid] + STDP_A_NEG)
        : (postsyn_exc_traces[nid] * STDP_TAU_NEG);
    time_since_spikes[nid] = (prev_bit)
        ? 0
        : MIN(32, time_since_spikes[nid] + 1);
    voltages[nid] = (spike) ? cs[nid] : voltage;
    recoveries[nid] = recovery + (spike * ds[nid]);
)

/******************************************************************************/
/************************* TRACE ACTIVATOR KERNELS ****************************/
/******************************************************************************/

// Extraction at start of kernel
#define ACTIV_EXTRACTIONS \
    SpnetAttributes *att = \
        (SpnetAttributes*)synapse_data.attributes; \
    SpnetWeightMatrix *matrix = \
        (SpnetWeightMatrix*)synapse_data.matrix; \
    float baseline_conductance = matrix->baseline_conductance; \
\
    int   *delays = matrix->delays.get(); \
    int   *from_time_since_spike = matrix->time_since_spike.get();

// Neuron Pre Operation
#define INIT_SUM \
    float sum = 0.0;

// Weight Operation
#define CALC_VAL \
    float spike = extract(outputs[from_index], delays[weight_index]); \
    from_time_since_spike[from_index] = \
        ((spike > 0.0) \
            ? 0 \
            : MIN(32, from_time_since_spike[from_index] + 1)); \
\
    float weight = weights[weight_index]; \
    sum += spike * weight * baseline_conductance;

// Neuron Post Operation
#define AGGREGATE \
    inputs[to_index] = aggregate(inputs[to_index], sum);

CALC_ALL(activate_spnet,
    ACTIV_EXTRACTIONS,

    INIT_SUM,

    CALC_VAL,

    AGGREGATE
);

Kernel<SYNAPSE_ARGS> SpnetAttributes::get_activator(Connection *conn) {
    // These are not supported because of the change of weight matrix pointer
    // Second order host connections require their weight matrices to be copied
    // Currently, this only copies the first matrix in the stack, and this
    //   attribute set uses auxiliary data
    if (conn->second_order)
        LOG_ERROR(
            "Unimplemented connection type!");

    return activate_spnet_map.at(conn->get_type());
}

/******************************************************************************/
/************************** TRACE UPDATER KERNELS *****************************/
/******************************************************************************/

#define UPDATE_EXTRACTIONS \
    SpnetWeightMatrix *matrix = \
        (SpnetWeightMatrix*)synapse_data.matrix; \
    float *presyn_traces   = matrix->presyn_traces.get(); \
    int   *delays = matrix->delays.get(); \
    int   *from_time_since_spike = matrix->time_since_spike.get(); \
    float *dws   = matrix->dw.get(); \
\
    SpnetAttributes *att = \
        (SpnetAttributes*)synapse_data.attributes; \
    float *to_exc_traces = att->postsyn_exc_trace.get(); \
    int   *to_time_since_spike = att->time_since_spike.get(); \
    float learning_rate = matrix->learning_rate; \

#define GET_DEST_ACTIVITY \
    float dest_exc_trace = to_exc_traces[to_index]; \
    int   dest_time_since_spike = to_time_since_spike[to_index]; \
    float dest_spike = extract(destination_outputs[to_index], 0);

/* Minimum weight */
#define MIN_WEIGHT 0.0001f

/* Time dynamics for long term eligibility trace */
#define C_TAU 0.99

#define UPDATE_WEIGHT \
    float weight = weights[weight_index]; \
\
    if (weight >= MIN_WEIGHT) { \
        /* Extract postsynaptic trace */ \
        int src_time_since_spike = from_time_since_spike[from_index]; \
        bool src_spike = src_time_since_spike == 0; \
\
        /* Update presynaptic trace */ \
        float src_trace = (opcode == ADD) \
            ? (presyn_traces[weight_index] = \
                (src_spike) \
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
            : ((src_spike and dest_time_since_spike > 0 and dest_time_since_spike < 32) \
                /* negative iSTDP function of delta T */ \
                ? powf(dest_time_since_spike, 10) \
                  * 0.0000001 \
                  * 2.6 \
                  * expf(-0.94 * dest_time_since_spike) \
                : 0.0); \
\
        /* Compute delta from short term dynamics */ \
        float dw = dws[weight_index] = \
            0.9 * dws[weight_index] + \
            (dest_spike * src_trace) \
            - (src_spike * dest_trace) ; \
        float weight_delta = dw + 0.01; \
\
        /* Calculate new weight */ \
        weight += learning_rate * weight_delta; \
\
        /* Ensure weight stays within boundaries */ \
        weights[weight_index] = MAX(MIN_WEIGHT, MIN(max_weight, weight)); \
    }

CALC_ALL(update_spnet,
    UPDATE_EXTRACTIONS,
    GET_DEST_ACTIVITY,
    UPDATE_WEIGHT,
);

Kernel<SYNAPSE_ARGS> SpnetAttributes::get_updater(Connection *conn) {
    // Second order, convolutional, and direct updaters are not supported
    if (conn->second_order or conn->convolutional)
        LOG_ERROR(
            "Unimplemented connection type!");

    try {
        switch (conn->opcode) {
            case(ADD):
            case(SUB):
                return update_spnet_map.at(conn->get_type());
        }
    } catch(std::out_of_range) { }

    LOG_ERROR(
        "Unimplemented connection type!");
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

    valid_params.insert("params");
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

SpnetAttributes::SpnetAttributes(Layer *layer)
        : Attributes(layer, BIT) {
    // Neuron variables
    this->voltage = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("voltage", &voltage);
    this->recovery = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("recovery", &recovery);
    this->postsyn_exc_trace = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("post trace", &postsyn_exc_trace);
    this->time_since_spike = Attributes::create_neuron_variable<int>();
    Attributes::register_neuron_variable("time since spike", &time_since_spike);

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

    create_parameters(layer->get_parameter("params", "regular"),
        this->as, this->bs, this->cs, this->ds, layer->size);
    for (int j = 0 ; j < layer->size ; ++j) {
        postsyn_exc_trace[j] = 0.0;
        time_since_spike[j] = 32;

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
        if (conn->get_type() == GAP and conn->from_layer != conn->to_layer)
            LOG_ERROR(
                "Error " + conn->str() + "\n"
                "Gap junctions must be between neurons of the same layer.");
    }
}

void SpnetWeightMatrix::register_variables() {
    // Presynaptic trace
    this->presyn_traces = WeightMatrix::create_variable<float>();
    WeightMatrix::register_variable("presyn trace", &presyn_traces);

    // Time since presynaptic spike
    this->time_since_spike = WeightMatrix::create_variable<int>();
    WeightMatrix::register_variable("time since spike", &time_since_spike);

    // Weight derivatives
    this->dw = WeightMatrix::create_variable<float>();
    WeightMatrix::register_variable("weight derivative", &dw);
}

void SpnetAttributes::process_weight_matrix(WeightMatrix* matrix) {
    auto iz_mat = (SpnetWeightMatrix*)matrix;
    Connection *conn = matrix->connection;

    // Retrieve baseline conductance
    iz_mat->baseline_conductance =
        std::stof(conn->get_parameter("conductance", "1.0"));

    // Retrieve learning rate
    iz_mat->learning_rate =
        std::stof(conn->get_parameter("learning rate", "0.1"));

    int num_weights = conn->get_num_weights();
    Pointer<float> mData = matrix->get_weights();

    // Plasticity trace
    fClear(iz_mat->presyn_traces, num_weights);

    // Weight derivatives
    fClear(iz_mat->dw, num_weights);

    // Delays
    // Myelinated connections use the base delay only
    if (conn->get_parameter("myelinated", "false") == "true") {
        iz_mat->get_delays(conn->delay);
    } else if (conn->get_config()->has("random delay")) {
        int max_delay = std::stoi(
            conn->get_parameter("random delay", "0"));
        if (max_delay > 31)
            LOG_ERROR(
                "Randomized axons cannot have delays greater than 31!");
        iz_mat->get_delays(0);
        iRand(iz_mat->delays, num_weights, 0, max_delay);
    } else {
        iz_mat->get_delays(BIT,
            std::stof(conn->from_layer->get_parameter("neuron spacing", "0.1")),
            std::stof(conn->to_layer->get_parameter("neuron spacing", "0.1")),
            std::stof(conn->get_parameter("x offset", "0.0")),
            std::stof(conn->get_parameter("y offset", "0.0")),
            0.15,
            conn->get_parameter("cap delay", "false") == "true");
    }

    // Time since last spike
    int *time_since_spike = iz_mat->time_since_spike.get();
    for (int i = 0 ; i < num_weights; ++i)
        time_since_spike[i] = 32;
}
