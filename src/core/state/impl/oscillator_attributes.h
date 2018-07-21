#ifndef oscillator_attributes_h
#define oscillator_attributes_h

#include "state/attributes.h"
#include "state/weight_matrix.h"

class OscillatorAttributes : public Attributes {
    public:
        OscillatorAttributes(Layer *layer);

        virtual bool check_compatibility(ClusterType cluster_type);

        virtual KernelList<SYNAPSE_ARGS> get_activators(Connection *conn);
        virtual KernelList<SYNAPSE_ARGS> get_updaters(Connection *conn);
        virtual KeySet get_init_keys() { return { "bold" }; }

        virtual void process_weight_matrix(WeightMatrix* matrix);

        // Internal neural state
        Pointer<float> state;

        // BOLD response (sum of absolute values of input)
        Pointer<float> bold;

        // Baseline tonic activity
        float tonic;

        // Time constant (scales synaptic input)
        float tau;

        // Decay time constant (scales decay)
        float decay;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

class OscillatorWeightMatrix : public WeightMatrix {
    WEIGHT_MATRIX_MEMBERS(OscillatorWeightMatrix);
    virtual void register_variables();
};

#endif
