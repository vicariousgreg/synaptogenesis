#ifndef hodgkin_huxley_attributes_h
#define hodgkin_huxley_attributes_h

#include "state/attributes.h"

/* Neuron parameters class.
 * Contains h,n,m parameters for Hodgkin-Huxley model */
class HodgkinHuxleyParameters {
    public:
        HodgkinHuxleyParameters(float iapp) : iapp(iapp) {}
        float iapp;
};

class HodgkinHuxleyAttributes : public Attributes {
    public:
        HodgkinHuxleyAttributes(Model* model);
        ~HodgkinHuxleyAttributes();

        virtual int get_matrix_depth() { return 3; }
        virtual void process_weight_matrix(WeightMatrix* matrix);

        virtual KERNEL get_activator(ConnectionType type) const {
            return get_activator_kernel_trace(type);
        }

        // Neuron Attributes
        float *voltage;
        float *h, *m, *n;
        float* current_trace;

        // Neuron Current (copy of input)
        float* current;

        // Neuron Spikes (copy of output)
        unsigned int* spikes;

        // Neuron parameters
        HodgkinHuxleyParameters* neuron_parameters;
};

#endif
