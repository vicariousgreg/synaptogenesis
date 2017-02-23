#ifndef attributes_h
#define attributes_h

#include <string>

#include "model/structure.h"
#include "state/weight_matrix.h"
#include "engine/kernel/extractor.h"
#include "engine/kernel/synapse_kernel.h"
#include "engine/kernel/activator_kernel.h"
#include "io/environment.h"
#include "util/constants.h"
#include "util/error_manager.h"

/* Typedef for attribute kernel functions */
typedef void(*ATTRIBUTE_KERNEL)(const Attributes*, int, int);

class Attributes {
    public:

        Attributes(Structure *structure, OutputType output_type);
        virtual ~Attributes();

        /* Gets the name of the stream cluster to use with these attributes */
        virtual std::string get_stream_cluster_name() { return "parallel"; }

        virtual void transfer_to_device();

        /* Attribute kernel getter */
        virtual ATTRIBUTE_KERNEL get_attribute_kernel() const = 0;

        /* Learning Rule functions */
        // Activator Kernel
        virtual SYNAPSE_KERNEL get_activator(ConnectionType type) {
            return get_base_activator_kernel(type);
        }

        // Updater Kernel
        virtual SYNAPSE_KERNEL get_updater(ConnectionType type) { return NULL; }

        // Depth of weight matrices
        virtual int get_matrix_depth(Connection *conn) { return 1; }

        // Weight matrix processor
        virtual void process_weight_matrix(WeightMatrix* matrix) { }

        // Layer data retrieval
        int get_start_index(int id) const;
        Pointer<float> get_input(int id) const;
        Pointer<Output> get_output(int id, int word_index = 0) const;

        // Number of neurons
        const int total_neurons;

        // Neuron IO data
        EXTRACTOR extractor;
        const OutputType output_type;
        Pointer<Output> output;
        Pointer<float> input;

        // Pointer to this object
        // If parallel, this will point to the device copy
        Attributes *pointer;

    protected:
        int max_input_registers;
        std::map<int, int> start_indices;
        std::map<int, int> sizes;
};

Attributes *build_attributes(Structure *structure);

#endif
