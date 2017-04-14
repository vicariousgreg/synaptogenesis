#ifndef spiking_attributes_h
#define spiking_attributes_h

#include "state/attributes.h"

class SpikingAttributes : public Attributes {
    public:
        SpikingAttributes(LayerList &layers, Kernel<ATTRIBUTE_ARGS> kernel);

        virtual Kernel<SYNAPSE_ARGS> get_activator(
            ConnectionType type, bool second_order);
        virtual Kernel<SYNAPSE_ARGS> get_updater(
            ConnectionType type, bool second_order);
        virtual int get_matrix_depth(Connection* conn) {
            return (conn->convolutional) ?
                (conn->to_layer->size + 2) : 3;
        }
        virtual void process_weight_matrix(WeightMatrix* matrix);

        // Neuron Attributes
        Pointer<float> voltage;
};

#endif
