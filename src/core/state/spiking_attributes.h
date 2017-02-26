#ifndef spiking_attributes_h
#define spiking_attributes_h

#include "state/attributes.h"

class SpikingAttributes : public Attributes {
    public:

        SpikingAttributes(Structure* structure, ATTRIBUTE_KERNEL kernel);
        virtual ~SpikingAttributes();

        /* Trace learning rules */
        virtual SYNAPSE_KERNEL get_activator(ConnectionType type);
        virtual SYNAPSE_KERNEL get_updater(ConnectionType type);
        virtual int get_matrix_depth(Connection* conn) {
            return (conn->convolutional) ?
                (conn->to_layer->size + 2) : 3;
        }
        virtual void process_weight_matrix(WeightMatrix* matrix);

        virtual void transfer_to_device();

        // Neuron Attributes
        Pointer<float> voltage;
};

#endif
