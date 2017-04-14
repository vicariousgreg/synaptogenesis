#include "state/backprop_rate_encoding_attributes.h"
#include "util/error_manager.h"

Kernel<SYNAPSE_ARGS> BackpropRateEncodingAttributes::get_updater(
        ConnectionType conn_type, bool second_order) {
    return Kernel<SYNAPSE_ARGS>(nullptr, nullptr);
}

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

BackpropRateEncodingAttributes::BackpropRateEncodingAttributes(LayerList &layers)
        : RateEncodingAttributes(layers) {
    this->error_deltas = Pointer<float>(total_neurons, 0.0);
    Attributes::register_variable(&this->error_deltas);
}
