#ifndef izhikevich_attributes_h
#define izhikevich_attributes_h

#include "state/attributes.h"

class SpnetAttributes : public Attributes {
    public:
        SpnetAttributes(Layer *layer);

        virtual KernelList<SYNAPSE_ARGS> get_activators(Connection *conn);
        virtual KernelList<SYNAPSE_ARGS> get_updaters(Connection *conn);
        virtual void process_weight_matrix(WeightMatrix* matrix);

        /* Neuron Attributes */

        // Voltage and recovery variables
        Pointer<float> voltage;
        Pointer<float> recovery;

        // Spike trace for learning
        Pointer<float> postsyn_exc_trace;
        Pointer<int> time_since_spike;

        // Neuron parameters
        Pointer<float> as;
        Pointer<float> bs;
        Pointer<float> cs;
        Pointer<float> ds;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

class SpnetWeightMatrix : public WeightMatrix {
    public:
        Pointer<float> presyn_traces;
        Pointer<int> time_since_spike;

        // Derivative of change
        Pointer<float> dw;

        // Baseline conductances
        float baseline_conductance;

        // Learning rate
        float learning_rate;

    WEIGHT_MATRIX_MEMBERS(SpnetWeightMatrix);
    virtual void register_variables();
};

#endif
