#ifndef izhikevich_attributes_h
#define izhikevich_attributes_h

#include "state/attributes.h"

/* Neuron parameters class.
 * Contains a,b,c,d parameters for Izhikevich model */
class IzhikevichParameters {
    public:
        IzhikevichParameters(float a, float b, float c, float d) :
                a(a), b(b), c(c), d(d) {}
        float a, b, c, d;
};

class IzhikevichAttributes : public Attributes {
    public:
        IzhikevichAttributes(LayerList &layers);

        virtual Kernel<SYNAPSE_ARGS> get_activator(
            Connection *conn, DendriticNode *node);
        virtual Kernel<SYNAPSE_ARGS> get_updater(
            Connection *conn, DendriticNode *node);
        virtual int get_matrix_depth(Connection* conn) {
            /*
             Weight
             Short term (AMPA/GABAA) conductance trace
             Long term (NMDA/GABAA) conductance trace
             Presynaptic trace
             Short Term Depression
             Short Term Potentiation
             Weight Delta
            */
            return 7;
        }
        virtual void process_weight_matrix(WeightMatrix* matrix);

        /* Connection Attributes */
        // Baseline conductances
        Pointer<float> baseline_conductance;

        // Learning rate
        Pointer<float> learning_rate;

        /* Neuron Attributes */
        // Conductances for different ion channels
        Pointer<float> ampa_conductance;
        Pointer<float> nmda_conductance;
        Pointer<float> gabaa_conductance;
        Pointer<float> gabab_conductance;

        // Multiplicative factor
        Pointer<float> multiplicative_factor;

        // Voltage and recovery variables
        Pointer<float> voltage;
        Pointer<float> recovery;

        // Spike trace for learning
        Pointer<float> postsyn_trace;
        // Time since last spike
        Pointer<int> delta_t;

        // Neuron parameters
        Pointer<IzhikevichParameters> neuron_parameters;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

#endif
