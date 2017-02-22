#ifndef attributes_h
#define attributes_h

#include "model/structure.h"
#include "state/weight_matrix.h"
#include "engine/kernel/kernel.h"
#include "engine/kernel/extractor.h"
#include "engine/kernel/activator_kernel.h"
#include "util/constants.h"
#include "util/error_manager.h"

/* Typedef for attribute kernel functions */
typedef void(*ATTRIBUTE_KERNEL)(const Attributes*, int, int);

class StreamCluster;

class Attributes {
    public:

        Attributes(Structure *structure, OutputType output_type);
        virtual ~Attributes();

        /* Builds an engine stream cluster based on attribute subclass
         * requirements. If none specified, uses a default parallel */
        virtual StreamCluster *build_stream_cluster(Structure *structure, State *state);

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

        // Layer data retrieval
        int get_start_index(int id) const;
        float* get_input(int id) const;
        Output* get_output(int id, int word_index = 0) const;

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

Attributes *build_attributes(Structure *structure);

#endif
