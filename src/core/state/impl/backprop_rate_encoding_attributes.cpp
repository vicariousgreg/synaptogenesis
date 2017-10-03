#include "state/impl/backprop_rate_encoding_attributes.h"
#include "util/error_manager.h"

REGISTER_ATTRIBUTES(BackpropRateEncodingAttributes, "backprop_rate_encoding", FLOAT)

Kernel<SYNAPSE_ARGS> BackpropRateEncodingAttributes::get_updater(
        ConnectionType conn_type, DendriticNode *node) {
    return Kernel<SYNAPSE_ARGS>(nullptr, nullptr);
}

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

BackpropRateEncodingAttributes::BackpropRateEncodingAttributes(LayerList &layers)
        : RateEncodingAttributes(layers) {
    this->error_deltas = Attributes::create_neuron_variable<float>(0.0);
    Attributes::register_neuron_variable("error deltas", &error_deltas);
}
