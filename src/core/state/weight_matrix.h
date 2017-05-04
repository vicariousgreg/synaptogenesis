#ifndef weight_matrix_h
#define weight_matrix_h

#include "util/constants.h"
#include "util/pointer.h"

class Connection;

class WeightMatrix {
    public:
        WeightMatrix(Connection *conn, int matrix_depth, DeviceID device_id);
        virtual ~WeightMatrix();

        Pointer<float> get_data() const { return mData; }

        void schedule_transfer();

        Connection* const connection;
        const DeviceID device_id;

    private:
        Pointer<float> mData;
        int matrix_size;
};

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

/* Transfers the values from one array to another */
void transfer_weights(float* from, float* to, int size);

/* Clears the diagonal of a weight matrix */
void clear_diagonal(float *arr, int rows, int cols);

/* Sets delays according to spatial organization */
void set_delays(Connection *conn, float* delays, float velocity,
    float from_spacing, float to_spacing, float x_offset, float y_offset);

#endif
