#ifndef attributes_h
#define attributes_h

#include <map>
#include <set>
#include <vector>

#include "model/layer.h"
#include "state/weight_matrix.h"
#include "engine/kernel/kernel.h"
#include "engine/kernel/synapse_kernel.h"
#include "engine/kernel/attribute_data.h"
#include "util/constants.h"
#include "util/pointer.h"
#include "util/error_manager.h"

/* Typedef for attribute kernel functions */
typedef AttributeData ATTRIBUTE_ARGS;
typedef void(*ATTRIBUTE_KERNEL)(ATTRIBUTE_ARGS);

/* Typedef for subclass build method
 * This can't be virtual because it's a class method */
typedef Attributes* (*BUILD_PTR)(LayerList &layers);

class Attributes {
    public:
        Attributes(LayerList &layers, OutputType output_type);
        virtual ~Attributes();

        /* Assigns the attributes to a device
         * This must be done prior to use */
        void set_device_id(DeviceID device_id);

        /* Checks whether these attributes are compatible
         *   with the given cluster_type */
        virtual bool check_compatibility(ClusterType cluster_type) {
            return true;
        }

        // Schedule and conduct transfer to device
        void schedule_transfer();
        void transfer_to_device();

        /* Learning Rule functions */
        // Activator Kernel
        virtual Kernel<SYNAPSE_ARGS> get_activator(
                ConnectionType type, bool second_order) {
            return get_base_activator_kernel(type, second_order);
        }

        // Updater Kernel
        virtual Kernel<SYNAPSE_ARGS> get_updater(
            ConnectionType type, bool second_order) {
            return Kernel<SYNAPSE_ARGS> ();
        }

        // Depth of weight matrices
        virtual int get_matrix_depth(Connection *conn) { return 1; }

        // Weight matrix processor
        virtual void process_weight_matrix(WeightMatrix* matrix) { }

        // Layer data retrieval
        int get_other_start_index(int id) const;
        Pointer<float> get_input(int id, int register_index = 0) const;
        Pointer<float> get_second_order_input(int id) const;
        Pointer<Output> get_output(int id, int word_index = 0) const;
        Pointer<Output> get_expected(int id) const;

        // Neuron IO data
        EXTRACTOR extractor;
        const OutputType output_type;
        Pointer<Output> output;
        Pointer<Output> expected;
        Pointer<float> input;
        Pointer<float> second_order_input;

        // Pointer to this object
        // If parallel, this will point to the device copy
        Attributes *pointer;

        // Getter for attribute kernels
        virtual Kernel<ATTRIBUTE_ARGS> get_kernel() = 0;
        virtual Kernel<ATTRIBUTE_ARGS> get_learning_kernel() {
            return Kernel<ATTRIBUTE_ARGS>(nullptr, nullptr);
        }

        DeviceID get_device_id() { return device_id; }

        // Get the set of neural model strings
        static const std::set<std::string> get_neural_models();

    protected:
        class NeuralModelBank {
            public:
                // Set of neural model implementations
                std::set<std::string> neural_models;
                std::map<std::string, BUILD_PTR> build_pointers;
                std::map<std::string, int> sizes;
        };

        friend Attributes *build_attributes(LayerList &layers,
            std::string neural_model, DeviceID device_id);

        // Registers a subclass neural model name with the state
        static int register_neural_model(std::string neural_model,
            int object_size, BUILD_PTR build_ptr);

        // Set of neural model implementations
        static NeuralModelBank* get_neural_model_bank();

        // Registers a variable to be handled by the superclass
        void register_variable(BasePointer *pointer);

        // Traverse the dendritic tree and find second order nodes
        int dendrite_DFS(DendriticNode *curr, int second_order_size);

        // Number of neurons and layers
        int total_neurons;
        int total_layers;

        DeviceID device_id;
        int object_size;

        // Managed pointers
        std::vector<BasePointer*> managed_variables;

        std::map<int, int> layer_indices;
        std::map<int, int> other_start_indices;
        std::map<int, int> input_start_indices;
        std::map<int, int> output_start_indices;
        std::map<int, int> expected_start_indices;
        std::map<int, int> layer_sizes;

        std::map<int, int> second_order_indices;
        std::map<int, int> second_order_sizes;
};

Attributes *build_attributes(LayerList &layers,
    std::string neural_model, DeviceID device_id);

/* Macros for Attribute subclass Registry */
// Put this one in .cpp
#define REGISTER_ATTRIBUTES(CLASS_NAME, STRING) \
int CLASS_NAME::neural_model_id = \
    Attributes::register_neural_model(STRING, \
        sizeof(CLASS_NAME), CLASS_NAME::build); \
\
Attributes *CLASS_NAME::build(LayerList &layers) { \
    return new CLASS_NAME(layers); \
}

// Put this one in .h at bottom of class definition
#define ATTRIBUTE_MEMBERS \
    private: \
        static Attributes *build(LayerList &layers); \
        static int neural_model_id;

// Put this one in .h if the class implements the attribute kernel
#define GET_KERNEL_DEF \
    public: \
        virtual Kernel<ATTRIBUTE_ARGS> get_kernel(); \

// Put this one in .h if the class implements the attribute learning kernel
#define GET_LEARNING_KERNEL_DEF \
    public: \
        virtual Kernel<ATTRIBUTE_ARGS> get_learning_kernel();



/* Macros for Attribute kernels */
#define PREAMBLE_ATTRIBUTES \
    const Attributes *att = attribute_data.attributes; \
    float *inputs = attribute_data.input.get(); \
    Output *outputs = attribute_data.output.get(); \
    int other_start_index = attribute_data.other_start_index; \
    int size = attribute_data.size; \
    int history_size = attribute_data.history_size; \
    bool plastic = attribute_data.plastic;

#ifdef __CUDACC__

// Skeletons -- don't use this directly
#define DEF_ATT_KERNEL(FUNC_NAME, PREAMBLE, BODY) \
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
}

// Use this to set up attributes kernel
#define BUILD_ATTRIBUTE_KERNEL( \
    CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
DEF_ATT_KERNEL(FUNC_NAME, PREAMBLE, BODY) \
Kernel<ATTRIBUTE_ARGS> CLASS_NAME::get_kernel() { \
    return Kernel<ATTRIBUTE_ARGS>(FUNC_NAME##_SERIAL, FUNC_NAME##_PARALLEL); \
}

// Use this to set up attributes learning kernel
#define BUILD_ATTRIBUTE_LEARNING_KERNEL( \
    CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
DEF_ATT_KERNEL(FUNC_NAME, PREAMBLE, BODY) \
Kernel<ATTRIBUTE_ARGS> CLASS_NAME::get_learning_kernel() { \
    return Kernel<ATTRIBUTE_ARGS>(FUNC_NAME##_SERIAL, FUNC_NAME##_PARALLEL); \
}

#else

// Skeletons -- don't use this directly
#define DEF_ATT_KERNEL(FUNC_NAME, PREAMBLE, BODY) \
GLOBAL void FUNC_NAME##_SERIAL(AttributeData attribute_data) { \
    PREAMBLE_ATTRIBUTES \
    PREAMBLE \
    for (int nid = 0; nid < size; ++nid) { \
        BODY; \
    } \
}

// Use this to set up attributes kernel
#define BUILD_ATTRIBUTE_KERNEL( \
    CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
DEF_ATT_KERNEL(FUNC_NAME, PREAMBLE, BODY) \
Kernel<ATTRIBUTE_ARGS> CLASS_NAME::get_kernel() { \
    return Kernel<ATTRIBUTE_ARGS>(FUNC_NAME##_SERIAL); \
}

// Use this to set up attributes learning kernel
#define BUILD_ATTRIBUTE_LEARNING_KERNEL( \
    CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
DEF_ATT_KERNEL(FUNC_NAME, PREAMBLE, BODY) \
Kernel<ATTRIBUTE_ARGS> CLASS_NAME::get_learning_kernel() { \
    return Kernel<ATTRIBUTE_ARGS>(FUNC_NAME##_SERIAL); \
}

#endif

#endif
