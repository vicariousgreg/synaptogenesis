#ifndef izhikevich_attributes_h
#define izhikevich_attributes_h

#include "state/attributes.h"

class IzhikevichAttributes : public Attributes {
    public:
        IzhikevichAttributes(LayerList &layers);

        virtual Kernel<SYNAPSE_ARGS> get_activator(Connection *conn);
        virtual Kernel<SYNAPSE_ARGS> get_updater(Connection *conn);
        virtual int get_matrix_depth(Connection* conn) {
            /*
             Weight
             Short term (AMPA/GABAA) conductance trace
             Long term (NMDA/GABAA) conductance trace
             Presynaptic trace
             Short Term Depression
             Short Term Potentiation
             Long term eligibility trace
             Delay
            */
            return 8;
        }
        virtual void process_weight_matrix(WeightMatrix* matrix);

        /* Connection Attributes */
        // Baseline conductances
        Pointer<float> baseline_conductance;

        // Learning rate
        Pointer<float> learning_rate;

        // Short term plasticity flag
        Pointer<int> stp_flag;

        /* Neuron Attributes */
        // Conductances for different ion channels
        Pointer<float> ampa_conductance;
        Pointer<float> nmda_conductance;
        Pointer<float> gabaa_conductance;
        Pointer<float> gabab_conductance;

        // Multiplicative factor
        Pointer<float> multiplicative_factor;

        // Reward signal
        Pointer<float> dopamine;

        // Modulation signal
        Pointer<float> acetylcholine;

        // Voltage and recovery variables
        Pointer<float> voltage;
        Pointer<float> recovery;

        // Spike trace for learning
        Pointer<float> postsyn_trace;

        // Neuron parameters
        Pointer<float> as;
        Pointer<float> bs;
        Pointer<float> cs;
        Pointer<float> ds;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
