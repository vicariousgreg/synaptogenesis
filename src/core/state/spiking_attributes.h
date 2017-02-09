#ifndef spiking_attributes_h
#define spiking_attributes_h

#include "state/attributes.h"

class SpikingAttributes : public Attributes {
    public:
        SpikingAttributes(Model* model);
        ~SpikingAttributes();

        virtual int get_matrix_depth() { return 3; }
        virtual void process_weight_matrix(WeightMatrix* matrix);

        virtual KERNEL get_activator(ConnectionType type) const {
            return get_activator_kernel_trace(type);
        }

#ifdef PARALLEL
        virtual void send_to_device();
#endif

        // Neuron Attributes
        float *voltage;

        // Neuron Current (copy of input)
        float* current;

        // Neuron Spikes (copy of output)
        unsigned int* spikes;
};

#endif
