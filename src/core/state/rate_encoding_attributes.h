#ifndef rate_encoding_attributes_h
#define rate_encoding_attributes_h

#include "state/state.h"

/* Neuron parameters class.
 * Contains parameters for Rate Encoding model */
class RateEncodingParameters {
    public:
        RateEncodingParameters(float x) : x(x) {}
        float x;
};

class RateEncodingAttributes : public Attributes {
    public:
        RateEncodingAttributes(Model* model);
        ~RateEncodingAttributes();

#ifdef PARALLEL
        virtual void send_to_device();
#endif

        virtual int get_matrix_depth() { return 1; }
        virtual void process_weight_matrix(WeightMatrix* matrix);

        virtual KERNEL get_activator(ConnectionType type) const {
            return get_activator_kernel(type);
        }

        virtual KERNEL get_updater(ConnectionType type) const {
            return NULL;
        }

        // Neuron parameters
        RateEncodingParameters* neuron_parameters;
};

#endif
