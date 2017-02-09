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

class Attributes {
    public:
        class LearningRule {
            public:
                // Activator Kernel
                virtual KERNEL get_activator(ConnectionType type) {
                    return get_base_activator_kernel(type);
                }

                // Updater Kernel
                virtual KERNEL get_updater(ConnectionType type) { return NULL; }

                // Depth of weight matrices
                virtual int get_matrix_depth() { return 1; }

                // Weight matrix processor
                virtual void process_weight_matrix(WeightMatrix* matrix) { }
        };

        Attributes(Model *model, OutputType output_type);
        virtual ~Attributes();

#ifdef PARALLEL
        virtual void send_to_device();
#endif

        /* Learning Rule getter */
        virtual LearningRule *get_learning_rule() {
            if (learning_rule == NULL)
                learning_rule = new Attributes::LearningRule();
            return learning_rule;
        }

        /* Attribute kernel getter */
        virtual ATTRIBUTE_KERNEL get_attribute_kernel() = 0;

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

        /* Primary attribute update function */
        ATTRIBUTE_KERNEL attribute_kernel;

        // Pointer to this object
        // If parallel, this will point to the device copy
        Attributes *pointer;

    protected:
        int num_neurons[sizeof(IOTypes)];
        int start_indices[sizeof(IOTypes)];
        int max_input_registers;
        LearningRule *learning_rule;
};

Attributes *build_attributes(Model *model);

#endif
