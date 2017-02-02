#ifndef attributes_h
#define attributes_h

#include "model/model.h"
#include "engine/kernel/kernel.h"
#include "engine/kernel/extractor.h"
#include "engine/kernel/updater_kernel.h"
#include "engine/kernel/attribute_kernel.h"
#include "util/constants.h"

class Attributes {
    public:
        Attributes(Model *model, OutputType output_type);
        virtual ~Attributes();

        // Depth of weight matrices
        virtual int get_matrix_depth() = 0;

        /* Constant getters */
        KERNEL get_updater(ConnectionType type) const { return get_updater_kernel(type); }
        int get_num_neurons(IOType type) const { return num_neurons[type]; }
        int get_start_index(IOType type) const { return start_indices[type]; }

        // Number of neurons
        const int total_neurons;

        // Neuron IO data
        EXTRACTOR extractor;
        const OutputType output_type;
        Output* recent_output;
        Output* output;
        float* input;

        /* Primary attribute update function */
        ATTRIBUTE_KERNEL attribute_kernel;

        // Pointer to this object
        // If parallel, this will point to the device copy
        Attributes *pointer;

    private:
        int num_neurons[sizeof(IOTypes)];
        int start_indices[sizeof(IOTypes)];
};

Attributes *build_attributes(Model *model);

#endif
