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

        KERNEL get_updater(ConnectionType type) { return get_updater_kernel(type); }

#ifdef PARALLEL
        // Pointer to device copy of this object
        Attributes *device_pointer;
#endif

        /* Primary attribute update function */
        ATTRIBUTE_KERNEL attribute_kernel;

        // Number of neurons
        int total_neurons;
        int num_neurons[IO_TYPE_SIZE];

        // Start indices by type
        int start_indices[IO_TYPE_SIZE];

        // Neuron input
        float* input;

        // Neuron output
        EXTRACTOR extractor;
        OutputType output_type;
        Output* output;
        Output* recent_output;
};

Attributes *build_attributes(Model *model);

#endif
