#ifndef attributes_h
#define attributes_h

#include <map>
#include <vector>

#include "network/layer.h"
#include "state/weight_matrix.h"
#include "state/neural_model_bank.h"
#include "engine/kernel/kernel.h"
#include "engine/kernel/synapse_data.h"
#include "engine/kernel/attribute_data.h"
#include "util/constants.h"
#include "util/pointer.h"
#include "util/error_manager.h"

class Attributes;

/* Typedef for attribute kernel functions */
typedef AttributeData ATTRIBUTE_ARGS;
typedef void(*ATTRIBUTE_KERNEL)(ATTRIBUTE_ARGS);

class Attributes {
    public:
        Attributes(Layer *layer, OutputType output_type);
        virtual ~Attributes();

        /* Checks whether these attributes are compatible
         *   with the given cluster_type */
        virtual bool check_compatibility(ClusterType cluster_type) {
            return true;
        }

        // Pointer sets and transfer functions
        std::vector<BasePointer*> get_pointers();
        std::map<PointerKey, BasePointer*> get_pointer_map();
        void transfer(DeviceID new_device);

        /* Learning Rule functions */
        // Activator Kernel
        virtual Kernel<SYNAPSE_ARGS> get_activator(Connection *conn) {
            return get_base_activator_kernel(conn);
        }

        // Updater Kernel
        virtual Kernel<SYNAPSE_ARGS> get_updater(Connection *conn) {
            return Kernel<SYNAPSE_ARGS> ();
        }

        // Weight matrix processor
        void process_weight_matrices();
        virtual void process_weight_matrix(WeightMatrix* matrix) { }
        void transpose_weight_matrices();
        WeightMatrix *get_weight_matrix(Connection *conn)
            { return weight_matrices.at(conn); }

        // Layer data retrieval
        Pointer<float> get_input(int register_index = 0) const;
        Pointer<Output> get_output(int word_index = 0) const;
        Pointer<Output> get_expected() const;

        // Getters for external use
        BasePointer* get_neuron_data(std::string key);

        // Neuron IO data
        const OutputType output_type;
        Pointer<Output> output;
        Pointer<Output> expected;
        Pointer<float> input;
        int input_register_count;
        int output_register_count;

        Layer * const layer;

        // Pointer to this object
        // If parallel, this will point to the device copy
        Attributes *pointer;

        // Getter for attribute kernels
        virtual Kernel<ATTRIBUTE_ARGS> get_kernel() = 0;
        virtual Kernel<ATTRIBUTE_ARGS> get_learning_kernel() {
            return Kernel<ATTRIBUTE_ARGS>(nullptr, nullptr);
        }

        DeviceID get_device_id() { return device_id; }

        // Gets the output type of a layer based on its neural model
        OutputType get_output_type();
        static OutputType get_output_type(std::string neural_model);
        static OutputType get_output_type(Layer *layer);

    protected:
        // Gets the name of the neural model
        virtual std::string get_neural_model() = 0;

        // Gets size of subclass object
        virtual int get_object_size() = 0;

        // Methods for creating and registering variables
        //   to be handled by the superclass
        template<class T> Pointer<T> create_neuron_variable();
        template<class T> Pointer<T> create_layer_variable();

        template<class T> Pointer<T> create_neuron_variable(T val);
        template<class T> Pointer<T> create_layer_variable(T val);

        void register_neuron_variable(std::string key, BasePointer* ptr);
        void register_layer_variable(std::string key, BasePointer* ptr);

        DeviceID device_id;

        // Managed pointers
        std::map<std::string, BasePointer*> neuron_variables;

        // Weight Matrices
        std::map<Connection*, WeightMatrix*> weight_matrices;
};

/* Macros for Attribute subclass Registry */
// Put this one in .cpp
#define REGISTER_ATTRIBUTES(CLASS_NAME, STRING, OUTPUT_TYPE) \
static bool __att_dummy = \
    NeuralModelBank::register_attributes( \
        STRING, OUTPUT_TYPE, CLASS_NAME::build); \
std::string CLASS_NAME::get_neural_model() {return STRING; } \
int CLASS_NAME::get_object_size() { return sizeof(CLASS_NAME); } \
\
Attributes *CLASS_NAME::build(Layer *layer) { \
    return new CLASS_NAME(layer); \
}

// Put this one in .h at bottom of class definition
#define ATTRIBUTE_MEMBERS \
    public: \
        static Attributes *build(Layer *layer); \
    protected: \
        virtual std::string get_neural_model(); \
        virtual int get_object_size();

// Put this one in .h if the class implements the attribute kernel
#define GET_KERNEL_DEF \
    public: \
        virtual Kernel<ATTRIBUTE_ARGS> get_kernel(); \

// Put this one in .h if the class implements the attribute learning kernel
#define GET_LEARNING_KERNEL_DEF \
    public: \
        virtual Kernel<ATTRIBUTE_ARGS> get_learning_kernel();



/* Macros for Attribute kernels */
#define PREAMBLE_ATTRIBUTES(CLASS_NAME) \
    CLASS_NAME *att = (CLASS_NAME*)attribute_data.attributes; \
    float *inputs = attribute_data.input.get(); \
    Output *outputs = attribute_data.output.get(); \
    int size = attribute_data.size; \
    int history_size = attribute_data.history_size; \
    bool plastic = attribute_data.plastic;

#ifdef __CUDACC__

// Skeletons -- don't use this directly
// Standard attribute kernel
#define DEF_ATT_KERNEL(CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
HOST void FUNC_NAME##_SERIAL(AttributeData attribute_data) { \
    PREAMBLE_ATTRIBUTES(CLASS_NAME) \
    PREAMBLE \
    for (int nid = 0; nid < size; ++nid) { \
        BODY; \
    } \
} \
GLOBAL void FUNC_NAME##_PARALLEL(AttributeData attribute_data) { \
    PREAMBLE_ATTRIBUTES(CLASS_NAME) \
    PREAMBLE \
    int nid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (nid < size) { \
        BODY; \
    } \
}

// Random attribute kernel
// Creates a random variables between 0.0 and 1.0
#define DEF_RAND_ATT_KERNEL(CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
HOST void FUNC_NAME##_SERIAL(AttributeData attribute_data) { \
    std::uniform_real_distribution<float> distribution(0.0, 1.0); \
    PREAMBLE_ATTRIBUTES(CLASS_NAME) \
    PREAMBLE \
    for (int nid = 0; nid < size; ++nid) { \
        float rand = distribution(generator); \
        BODY; \
    } \
} \
GLOBAL void FUNC_NAME##_PARALLEL(AttributeData attribute_data) { \
    PREAMBLE_ATTRIBUTES(CLASS_NAME) \
    PREAMBLE \
    int nid = blockIdx.x * blockDim.x + threadIdx.x; \
    if (nid < size) { \
        float rand = curand_uniform(&cuda_rand_states[nid]); \
        BODY; \
    } \
}

// Use this to set up attributes kernel
// Standard version
#define BUILD_ATTRIBUTE_KERNEL( \
    CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
DEF_ATT_KERNEL(CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
Kernel<ATTRIBUTE_ARGS> CLASS_NAME::get_kernel() { \
    return Kernel<ATTRIBUTE_ARGS>(FUNC_NAME##_SERIAL, FUNC_NAME##_PARALLEL); \
}

// Random version
#define BUILD_RAND_ATTRIBUTE_KERNEL( \
    CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
DEF_RAND_ATT_KERNEL(CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
Kernel<ATTRIBUTE_ARGS> CLASS_NAME::get_kernel() { \
    return Kernel<ATTRIBUTE_ARGS>(FUNC_NAME##_SERIAL, FUNC_NAME##_PARALLEL); \
}

// Use this to set up attributes learning kernel
#define BUILD_ATTRIBUTE_LEARNING_KERNEL( \
    CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
DEF_ATT_KERNEL(CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
Kernel<ATTRIBUTE_ARGS> CLASS_NAME::get_learning_kernel() { \
    return Kernel<ATTRIBUTE_ARGS>(FUNC_NAME##_SERIAL, FUNC_NAME##_PARALLEL); \
}

#else

// Skeletons -- don't use this directly
#define DEF_ATT_KERNEL(CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
HOST void FUNC_NAME##_SERIAL(AttributeData attribute_data) { \
    PREAMBLE_ATTRIBUTES(CLASS_NAME) \
    PREAMBLE \
    for (int nid = 0; nid < size; ++nid) { \
        BODY; \
    } \
}

// Random attribute kernel
// Creates a random variables between 0.0 and 1.0
#define DEF_RAND_ATT_KERNEL(CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
HOST void FUNC_NAME##_SERIAL(AttributeData attribute_data) { \
    std::uniform_real_distribution<float> distribution(0.0, 1.0); \
    PREAMBLE_ATTRIBUTES(CLASS_NAME) \
    PREAMBLE \
    for (int nid = 0; nid < size; ++nid) { \
        float rand = distribution(generator); \
        BODY; \
    } \
}

// Use this to set up attributes kernel
#define BUILD_ATTRIBUTE_KERNEL( \
    CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
DEF_ATT_KERNEL(CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
Kernel<ATTRIBUTE_ARGS> CLASS_NAME::get_kernel() { \
    return Kernel<ATTRIBUTE_ARGS>(FUNC_NAME##_SERIAL); \
}

// Random version
#define BUILD_RAND_ATTRIBUTE_KERNEL( \
    CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
DEF_RAND_ATT_KERNEL(CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
Kernel<ATTRIBUTE_ARGS> CLASS_NAME::get_kernel() { \
    return Kernel<ATTRIBUTE_ARGS>(FUNC_NAME##_SERIAL); \
}

// Use this to set up attributes learning kernel
#define BUILD_ATTRIBUTE_LEARNING_KERNEL( \
    CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
DEF_ATT_KERNEL(CLASS_NAME, FUNC_NAME, PREAMBLE, BODY) \
Kernel<ATTRIBUTE_ARGS> CLASS_NAME::get_learning_kernel() { \
    return Kernel<ATTRIBUTE_ARGS>(FUNC_NAME##_SERIAL); \
}

#endif

// Macros for shifting outputs
#define SHIFT_FLOAT_OUTPUTS(f_outputs, new_output) \
    for (int index = history_size - 1 ; index > 0 ; --index) \
        f_outputs[size * index + nid] = f_outputs[size * (index - 1) + nid]; \
    f_outputs[nid] = new_output;

#define SHIFT_BIT_OUTPUTS(b_outputs, new_bit) \
    /* Reduce reads, chain values */ \
    unsigned int curr_value = b_outputs[size * (history_size-1) + nid]; \
\
    for (int index = history_size - 1 ; index > 0 ; --index) { \
        unsigned int prev_value = b_outputs[size * (index-1) + nid]; \
        /* Shift bits, carry over LSB from prev value. */ \
        b_outputs[size * index + nid] = (curr_value >> 1) | (prev_value << 31); \
        curr_value = prev_value; \
    } \
\
    b_outputs[nid] = (curr_value >> 1) | (new_bit << 31); \
    bool prev_bit = curr_value >> 31;

#endif
