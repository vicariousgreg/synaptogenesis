#ifndef oscillator_attributes_h
#define oscillator_attributes_h

#include "state/attributes.h"
#include "state/weight_matrix.h"

class NVMAttributes : public Attributes {
    public:
        NVMAttributes(Layer *layer);

        virtual KernelList<SYNAPSE_ARGS> get_activators(Connection *conn);

        virtual void process_weight_matrix(WeightMatrix* matrix);

        // Internal neural state
        Pointer<float> state;

        // Gate registers
        float activity_gate, learning_gate;

        // Normalization factor for learning
        float norm;

        // Noise parameters
        // noise = rand * noise_scale - noise_offset
        float noise_scale;
        float noise_offset;

        // Continuous gate factor variables
        float pad;
        float ohr;

    GET_KERNEL_DEF
    ATTRIBUTE_MEMBERS
};

class NVMWeightMatrix : public WeightMatrix {
    public:
        float norm;

    WEIGHT_MATRIX_MEMBERS(NVMWeightMatrix);
};

#endif
