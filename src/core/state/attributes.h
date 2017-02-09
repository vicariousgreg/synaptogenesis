#ifndef attributes_h
#define attributes_h

#include "model/model.h"
#include "state/weight_matrix.h"
#include "engine/kernel/kernel.h"
#include "engine/kernel/extractor.h"
#include "engine/kernel/activator_kernel.h"
#include "engine/kernel/updater_kernel.h"
#include "engine/kernel/attribute_kernel.h"
#include "util/constants.h"

class Attributes {
    public:
        Attributes(Model *model, OutputType output_type);
        virtual ~Attributes();

#ifdef PARALLEL
        virtual void send_to_device();
#endif

        // Depth of weight matrices
        virtual int get_matrix_depth() = 0;

        // Weight matrix processor
        virtual void process_weight_matrix(WeightMatrix* matrix) { }

        /* Constant getters */
        virtual KERNEL get_activator(ConnectionType type) const {
            return get_activator_kernel(type);
        }
        virtual KERNEL get_updater(ConnectionType type) const {
            return get_updater_kernel(type);
        }
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
        int max_input_registers;
};

Attributes *build_attributes(Model *model);

#endif
