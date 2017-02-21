#ifndef attributes_h
#define attributes_h

#include "model/model.h"
#include "state/weight_matrix.h"
#include "engine/kernel/kernel.h"
#include "engine/kernel/extractor.h"
#include "engine/kernel/activator_kernel.h"
#include "util/constants.h"
#include "util/error_manager.h"

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
        virtual ATTRIBUTE_KERNEL get_attribute_kernel() const = 0;

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

        int get_start_index(int id) const { return start_indices.at(id); }
        float* get_input(int id) const { return input + start_indices.at(id); }
        Output* get_output(int id, int word_index = 0) const {
            if (word_index >= HISTORY_SIZE)
                ErrorManager::get_instance()->log_error(
                    "Cannot retrieve output word index past history length!");
            return output + (total_neurons * word_index) + start_indices.at(id);
        }

        // Number of neurons
        const int total_neurons;

        // Neuron IO data
        EXTRACTOR extractor;
        const OutputType output_type;
        Output* output;
        float* input;

        // Pointer to this object
        // If parallel, this will point to the device copy
        Attributes *pointer;

    protected:
        int max_input_registers;
        std::map<int, int> start_indices;
};

Attributes *build_attributes(Model *model);

#endif
