#include "state/backprop_rate_encoding_attributes.h"
#include "util/error_manager.h"

Kernel<SYNAPSE_ARGS> BackpropRateEncodingAttributes::get_updater(ConnectionType conn_type) {
    return Kernel<SYNAPSE_ARGS>(nullptr, nullptr);
}

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

BackpropRateEncodingAttributes::BackpropRateEncodingAttributes(LayerList &layers)
        : RateEncodingAttributes(layers) {
    this->error_deltas = Pointer<float>(total_neurons, 0.0);
}

BackpropRateEncodingAttributes::~BackpropRateEncodingAttributes() {
    this->error_deltas.free();
}

void BackpropRateEncodingAttributes::schedule_transfer() {
    RateEncodingAttributes::schedule_transfer();

    this->error_deltas.schedule_transfer(device_id);
}

