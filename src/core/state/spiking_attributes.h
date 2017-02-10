#ifndef spiking_attributes_h
#define spiking_attributes_h

#include "state/attributes.h"

class SpikingAttributes : public Attributes {
    public:

        SpikingAttributes(Model* model);
        virtual ~SpikingAttributes();

        /* Trace learning rules */
        virtual KERNEL get_activator(ConnectionType type);
        virtual KERNEL get_updater(ConnectionType type);
        virtual int get_matrix_depth(Connection* conn) {
            return (conn->convolutional) ?
                (conn->to_layer->size + 2) : 3;
        }
        virtual void process_weight_matrix(WeightMatrix* matrix);

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
