#ifndef spiking_attributes_h
#define spiking_attributes_h

#include "state/attributes.h"

class SpikingAttributes : public Attributes {
    public:
        SpikingAttributes(LayerList &layers);

        virtual Kernel<SYNAPSE_ARGS> get_activator(
            ConnectionType type, bool second_order);
        virtual Kernel<SYNAPSE_ARGS> get_updater(
            ConnectionType type, bool second_order);
        virtual int get_matrix_depth(Connection* conn) {
            /*
             Weight
             Baseline
             Short term (AMPA/GABAA) conductance trace
             Long term (NMDA/GABAB) conductance trace
             Plasticity trace
            */
            return 5;
        }
        virtual void process_weight_matrix(WeightMatrix* matrix);

        // Neuron Attributes
        Pointer<float> voltage;
        Pointer<float> neuron_trace;
};

#endif
