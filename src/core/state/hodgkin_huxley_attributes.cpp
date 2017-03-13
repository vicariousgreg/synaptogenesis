#include <string>
#include <sstream>

#include "state/hodgkin_huxley_attributes.h"
#include "util/tools.h"
#include "util/error_manager.h"

/******************************************************************************/
/******************************** PARAMS **************************************/
/******************************************************************************/

static HodgkinHuxleyParameters create_parameters(std::string str) {
    std::stringstream stream(str);
    float iapp;
    stream >> iapp;
    return HodgkinHuxleyParameters(iapp);
}

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

#include <math.h>

/* Voltage threshold for neuron spiking. */
#define HH_SPIKE_THRESH 30.0

/* Euler resolution for voltage update. */
#define HH_RESOLUTION 30
#define HH_TIMESTEPS 30

#define HH_GNABAR 120.0
#define HH_VNA 50.0
#define HH_GKBAR 36.0
#define HH_VK -77.0
#define HH_GL 0.3
#define HH_VL -54.4
#define HH_CM 1.0

BUILD_ATTRIBUTE_KERNEL(hh_attribute_kernel,
    HodgkinHuxleyAttributes *hh_att = (HodgkinHuxleyAttributes*)att;
    float *voltages = hh_att->voltage.get(other_start_index);
    float *hs = hh_att->h.get(other_start_index);
    float *ms = hh_att->m.get(other_start_index);
    float *ns = hh_att->n.get(other_start_index);
    float *current_traces = hh_att->current_trace.get(other_start_index);
    unsigned int *spikes = (unsigned int*)outputs;
    HodgkinHuxleyParameters *params = hh_att->neuron_parameters.get(other_start_index);

    ,

    /**********************
     *** VOLTAGE UPDATE ***
     **********************/
    float current =
         (inputs[nid] / HH_RESOLUTION) +
        (current_traces[nid] * 0.1);

    float voltage = voltages[nid];
    float h = hs[nid];
    float m = ms[nid];
    float n = ns[nid];
    float iapp = params[nid].iapp;

    // Euler's method for voltage/recovery update
    // If the voltage exceeds the spiking threshold, break
    bool already_spiked = voltage > HH_SPIKE_THRESH;
    unsigned int spike = 0;
    for (int i = 0 ; i < HH_TIMESTEPS ; ++i) {
        if (not spike and not already_spiked) m += current / HH_RESOLUTION;
        float am   = 0.1*(voltage+40.0)/( 1.0 - expf(-(voltage+40.0)/10.0) );
        float bm   = 4.0*expf(-(voltage+65.0)/18.0);
        float minf = am/(am+bm);
        float taum = 1.0/(am+bm);

        float ah   = 0.07*expf(-(voltage+65.0)/20.0);
        float bh   = 1.0/( 1.0 + expf(-(voltage+35.0)/10.0) );
        float hinf = ah/(ah+bh);
        float tauh = 1.0/(ah+bh);

        float an   = 0.01*(voltage + 55.0)/(1.0 - expf(-(voltage + 55.0)/10.0));
        float bn   = 0.125*expf(-(voltage + 65.0)/80.0);
        float ninf = an/(an+bn);
        float taun = 1.0/(an+bn);

        float ina = HH_GNABAR * (m*m*m) * h * (voltage-HH_VNA);
        float ik  = HH_GKBAR * (n*n*n*n) * (voltage-HH_VK);
        float il  = HH_GL * (voltage-HH_VL);

        voltage += (iapp - ina - ik - il ) / (HH_RESOLUTION * HH_CM);
        if (voltage != voltage) voltage = HH_SPIKE_THRESH+1;
        else {
            h +=  (hinf - h) / (HH_RESOLUTION * tauh);
            n +=  (ninf - n) / (HH_RESOLUTION * taun);
            m +=  (minf - m) / (HH_RESOLUTION * taum);
        }

        spike = spike or voltage > HH_SPIKE_THRESH;
    }
    spike = spike and not already_spiked;

    hs[nid] = h;
    ms[nid] = m;
    ns[nid] = n;
    voltages[nid] = voltage;
    current_traces[nid] = current;

    /********************
     *** SPIKE UPDATE ***
     ********************/
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
)

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

HodgkinHuxleyAttributes::HodgkinHuxleyAttributes(LayerList &layers)
        : SpikingAttributes(layers, get_hh_attribute_kernel()) {
    this->h = Pointer<float>(total_neurons);
    this->m = Pointer<float>(total_neurons);
    this->n = Pointer<float>(total_neurons);
    this->current_trace = Pointer<float>(total_neurons);
    this->neuron_parameters = Pointer<HodgkinHuxleyParameters>(total_neurons);

    // Fill in table
    int start_index = 0;
    for (auto& layer : layers) {
        HodgkinHuxleyParameters params = create_parameters(layer->params);
        for (int j = 0 ; j < layer->size ; ++j) {
            neuron_parameters[start_index+j] = params;
            voltage[start_index+j] = -64.9997224337;
            h[start_index+j] = 0.596111046355;
            m[start_index+j] = 0.0529342176209;
            n[start_index+j] = 0.31768116758;
            current_trace[start_index+j] = 0.0;
        }
        start_index += layer->size;
    }
}

HodgkinHuxleyAttributes::~HodgkinHuxleyAttributes() {
    this->h.free();
    this->m.free();
    this->n.free();
    this->current_trace.free();
    this->neuron_parameters.free();
}

void HodgkinHuxleyAttributes::schedule_transfer() {
    SpikingAttributes::schedule_transfer();

    this->h.schedule_transfer(device_id);
    this->m.schedule_transfer(device_id);
    this->n.schedule_transfer(device_id);
    this->current_trace.schedule_transfer(device_id);
    this->neuron_parameters.schedule_transfer(device_id);
}
