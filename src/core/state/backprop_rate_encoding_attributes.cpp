#include "state/backprop_rate_encoding_attributes.h"
#include "util/error_manager.h"

int BackpropRateEncodingAttributes::neural_model_id =
    Attributes::register_neural_model("backprop_rate_encoding",
        sizeof(BackpropRateEncodingAttributes), BackpropRateEncodingAttributes::build);

Attributes *BackpropRateEncodingAttributes::build(LayerList &layers) {
    return new BackpropRateEncodingAttributes(layers);
}

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
