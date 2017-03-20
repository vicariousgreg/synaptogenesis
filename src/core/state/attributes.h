#ifndef attributes_h
#define attributes_h

#include <map>

#include "model/layer.h"
#include "state/weight_matrix.h"
#include "engine/kernel/kernel.h"
#include "engine/kernel/synapse_kernel.h"
#include "engine/kernel/attribute_data.h"
#include "util/constants.h"
#include "util/error_manager.h"

/* Typedef for attribute kernel functions */
typedef AttributeData ATTRIBUTE_ARGS;
typedef void(*ATTRIBUTE_KERNEL)(ATTRIBUTE_ARGS);

class Attributes {
    public:
        Attributes(LayerList &layers, OutputType output_type,
            Kernel<ATTRIBUTE_ARGS> kernel,
            Kernel<ATTRIBUTE_ARGS> learning_kernel=Kernel<ATTRIBUTE_ARGS>());
        virtual ~Attributes();

        void set_device_id(DeviceID device_id) {
            this->device_id = device_id;

            // Retrieve extractor
            // This has to wait until device_id is set
            get_extractor(&this->extractor, output_type, device_id);
        }

        /* Checks whether these attributes are compatible
         *   with the given cluster_type */
        virtual bool check_compatibility(ClusterType cluster_type) { return true; }

        virtual void schedule_transfer();
        void transfer_to_device();

        /* Learning Rule functions */
        // Activator Kernel
        virtual Kernel<SYNAPSE_ARGS> get_activator(ConnectionType type) {
            return get_base_activator_kernel(type);
        }

        // Updater Kernel
        virtual Kernel<SYNAPSE_ARGS> get_updater(ConnectionType type) {
            return Kernel<SYNAPSE_ARGS> ();
        }

        // Depth of weight matrices
        virtual int get_matrix_depth(Connection *conn) { return 1; }

        // Weight matrix processor
        virtual void process_weight_matrix(WeightMatrix* matrix) { }

        // Layer data retrieval
        int get_other_start_index(int id) const;
        Pointer<float> get_input(int id, int register_index = 0) const;
        Pointer<Output> get_output(int id, int word_index = 0) const;
        Pointer<Output> get_expected(int id) const;

        // Neuron IO data
        EXTRACTOR extractor;
        const OutputType output_type;
        Pointer<Output> output;
        Pointer<Output> expected;
        Pointer<float> input;

        // Pointer to this object
        // If parallel, this will point to the device copy
        Attributes *pointer;

        // Pointer to attribute kernel
        Kernel<ATTRIBUTE_ARGS> kernel;
        Kernel<ATTRIBUTE_ARGS> learning_kernel;

        DeviceID get_device_id() { return device_id; }


    protected:
        friend Attributes *build_attributes(LayerList &layers,
            NeuralModel neural_model, DeviceID device_id);

        // Number of neurons
        int total_neurons;

        DeviceID device_id;
        int object_size;

        std::map<int, int> other_start_indices;
        std::map<int, int> input_start_indices;
        std::map<int, int> output_start_indices;
        std::map<int, int> expected_start_indices;
        std::map<int, int> sizes;
};

Attributes *build_attributes(LayerList &layers,
    NeuralModel neural_model, DeviceID device_id);

#define PREAMBLE_ATTRIBUTES \
    const Attributes *att = attribute_data.attributes; \
    float *inputs = attribute_data.input.get(); \
    Output *outputs = attribute_data.output.get(); \
    int other_start_index = attribute_data.other_start_index; \
    int size = attribute_data.size; \
    int history_size = attribute_data.history_size; \
    bool plastic = attribute_data.plastic;

#ifdef __CUDACC__

#define BUILD_ATTRIBUTE_KERNEL( \
    FUNC_NAME, PREAMBLE, BODY) \
GLOBAL void FUNC_NAME##_SERIAL(AttributeData attribute_data) { \
    PREAMBLE_ATTRIBUTES \
    PREAMBLE \
    for (int nid = 0; nid < size; ++nid) { \
        BODY; \
    } \
} \
GLOBAL void FUNC_NAME##_PARALLEL(AttributeData attribute_data) { \
    PREAMBLE_ATTRIBUTES \
    PREAMBLE \
    int nid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (nid < size) { \
        BODY; \
    } \
} \
static Kernel<ATTRIBUTE_ARGS> get_##FUNC_NAME() { \
    return Kernel<ATTRIBUTE_ARGS>(FUNC_NAME##_SERIAL, FUNC_NAME##_PARALLEL); \
}

#else

#define BUILD_ATTRIBUTE_KERNEL( \
    FUNC_NAME, PREAMBLE, BODY) \
GLOBAL void FUNC_NAME##_SERIAL(AttributeData attribute_data) { \
    PREAMBLE_ATTRIBUTES \
    PREAMBLE \
    for (int nid = 0; nid < size; ++nid) { \
        BODY; \
    } \
} \
static Kernel<ATTRIBUTE_ARGS> get_##FUNC_NAME() { \
    return Kernel<ATTRIBUTE_ARGS>(FUNC_NAME##_SERIAL); \
}

#endif

#endif
