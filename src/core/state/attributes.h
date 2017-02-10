#ifndef attributes_h
#define attributes_h

#include "model/model.h"
#include "state/weight_matrix.h"
#include "engine/kernel/kernel.h"
#include "engine/kernel/extractor.h"
#include "engine/kernel/activator_kernel.h"
#include "util/constants.h"

/* Typedef for attribute kernel functions */
typedef void(*ATTRIBUTE_KERNEL)(const Attributes*, int, int);

class Engine;

class Attributes {
    public:

        Attributes(Model *model, OutputType output_type);
        virtual ~Attributes();

        /* Builds an engine based on attribute subclass requirements
         * If none specified, uses a default parallel engine */
        virtual Engine *build_engine(Model *model, State *state);

#ifdef PARALLEL
        virtual void send_to_device();
#endif
        /* Attribute kernel getter */
        virtual ATTRIBUTE_KERNEL get_attribute_kernel() = 0;

        /* Learning Rule functions */
        // Activator Kernel
        virtual KERNEL get_activator(ConnectionType type) {
            return get_base_activator_kernel(type);
        }

        // Updater Kernel
        virtual KERNEL get_updater(ConnectionType type) { return NULL; }

        // Depth of weight matrices
        virtual int get_matrix_depth(Connection *conn) { return 1; }

        // Weight matrix processor
        virtual void process_weight_matrix(WeightMatrix* matrix) { }

        /* Constant getters */
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

        // Pointer to this object
        // If parallel, this will point to the device copy
        Attributes *pointer;

    protected:
        int num_neurons[sizeof(IOTypes)];
        int start_indices[sizeof(IOTypes)];
        int max_input_registers;
};

Attributes *build_attributes(Model *model);

#endif
