#include "state/spiking_attributes.h"
#include "util/tools.h"
#include "util/error_manager.h"
#include "util/parallel.h"

SpikingAttributes::SpikingAttributes(Model* model) : Attributes(model, BIT) {
    this->voltage = (float*) allocate_host(total_neurons, sizeof(float));
    this->current = this->input;
    this->spikes = (unsigned int*)this->output;
}

SpikingAttributes::~SpikingAttributes() {
#ifdef PARALLEL
    cudaFree(this->voltage);
#else
    free(this->voltage);
#endif
}

#ifdef PARALLEL
void SpikingAttributes::send_to_device() {
    Attributes::send_to_device();

    // Update these
    this->current = this->input;
    this->spikes = (unsigned int*)this->output;

    // Allocate space on GPU and copy data
    float *device_voltage = (float*)
        allocate_device(total_neurons, sizeof(float), this->voltage);

    free(this->voltage);
    this->voltage = device_voltage;
}
#endif

void SpikingAttributes::process_weight_matrix(WeightMatrix* matrix) {
    Connection *conn = matrix->connection;
    float *mData = matrix->get_data();
    if (conn->plastic) {
        int num_weights = conn->get_num_weights();

        // Baseline
        transfer_weights(mData, mData + num_weights, num_weights);

        // Trace
        clear_weights(mData + 2*num_weights, num_weights);
    }
}
