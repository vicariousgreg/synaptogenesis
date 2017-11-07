#ifndef weight_matrix_h
#define weight_matrix_h

#include "util/constants.h"
#include "util/pointer.h"

class Connection;
class Attributes;

class WeightMatrix {
    public:
        WeightMatrix(Attributes *att, Connection *conn);
        virtual ~WeightMatrix();

        // Getters
        Pointer<float> get_weights() const { return weights; }
        Pointer<float> get_second_order_weights() const
            { return second_order_weights; }
        BasePointer* get_layer(std::string key);
        DeviceID get_device_id() { return device_id; }

        // Pointer sets and transfer functions
        std::vector<BasePointer*> get_pointers();
        std::map<PointerKey, BasePointer*> get_pointer_map();
        void transpose(bool to_device);
        void transfer_to_device();
        void transfer_to_host();

        // Subclasses implement this for variable registration
        virtual void register_variables() { }

        // Pointer to this object
        // If parallel, this will point to the device copy
        WeightMatrix *pointer;

        Attributes* const attributes;
        Connection* const connection;

        // Build function
        static WeightMatrix *build(Attributes *att,
            Connection *conn, DeviceID device_id);

    protected:
        // Additional matrix layers (variables)
        template<class T> Pointer<T> create_variable();
        void register_variable(std::string key, BasePointer* ptr);
        std::map<std::string, BasePointer*> variables;

        Pointer<float> weights;
        Pointer<float> second_order_weights;
        int num_weights;
        DeviceID device_id;

        // Initialization
        void init(DeviceID device_id);

        virtual int get_object_size() { return sizeof(WeightMatrix); }
};

/* Macros for WeightMatrix subclass Registry */
// Put this one in .cpp
#define REGISTER_WEIGHT_MATRIX(CLASS_NAME, STRING) \
CLASS_NAME::CLASS_NAME(Attributes *att, Connection *conn) \
    : WeightMatrix(att, conn) { } \
static bool __mat_dummy = \
    NeuralModelBank::register_weight_matrix( \
        STRING, CLASS_NAME::build); \
int CLASS_NAME::get_object_size() { return sizeof(CLASS_NAME); } \
\
WeightMatrix *CLASS_NAME::build(Attributes *att, Connection *conn, DeviceID device_id) { \
    auto mat = new CLASS_NAME(att, conn); \
    mat->init(device_id); \
    return mat; \
}

// Put this one in .h at bottom of class definition
#define WEIGHT_MATRIX_MEMBERS(CLASS_NAME) \
    public: \
        CLASS_NAME(Attributes *att, Connection *conn); \
        static WeightMatrix *build(Attributes *att, Connection *conn, DeviceID device_id); \
    protected: \
        virtual int get_object_size();

/* Sets all values in an array to the given val */
void set_weights(float* arr, int size, float val, float fraction=1.0);

/* Clears an array */
void clear_weights(float* arr, int size);

/* Randomizes an array */
void randomize_weights(float* arr, int size, float max, float fraction=1.0);
void randomize_weights_gaussian(float* arr, int size,
    float mean, float std_dev, float max, float fraction=1.0);
void randomize_weights_lognormal(float* arr, int size,
    float mean, float std_dev, float max, float fraction=1.0);
void randomize_weights_powerlaw(float* arr, int size,
    float exponent, float max, float fraction=1.0);

/* Transfers the values from one array to another */
void transfer_weights(float* from, float* to, int size);

/* Clears the diagonal of a weight matrix */
void clear_diagonal(float *arr, int rows, int cols);

/* Sets delays according to spatial organization */
void set_delays(DeviceID device_id, OutputType output_type, Connection *conn,
    int* delays, float velocity, bool cap_delay,
    float from_spacing, float to_spacing,
    float x_offset, float y_offset);

/* Transposes a matrix in place */
template <typename T>
void transpose_matrix(T* data, int original_rows, int original_cols);

#endif
