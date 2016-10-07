#include <cstdlib>
#include <sstream>

#include "state/state.h"
#include "tools.h"
#include "parallel.h"

float* State::get_input() {
#ifdef PARALLEL
    // Copy from GPU to local location
    cudaMemcpy(this->input, this->device_input,
        this->model->num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("Failed to copy input from device to host!");
#endif
    return this->input;
}

void* State::get_output() {
#ifdef PARALLEL
    // Copy from GPU to local location
    cudaMemcpy(this->output, this->device_output,
        this->model->num_neurons * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError("Failed to copy output from device to host!");
#endif
    return this->output;
}

void State::set_input(int layer_id, float* input) {
    int offset = this->model->layers[layer_id]->index;
    int size = this->model->layers[layer_id]->size;
#ifdef PARALLEL
    // Send to GPU
    cudaMemcpy(&this->device_input[offset], input,
        size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError("Failed to set input!");
#else
    for (int nid = 0 ; nid < size; ++nid) {
        this->input[nid+offset] = input[nid];
    }
#endif
}

void State::clear_all_input() {
    for (int nid = 0 ; nid < this->model->num_neurons; ++nid)
        this->input[nid] = 0.0;

#ifdef PARALLEL
    // Send to GPU
    cudaMemcpy(this->device_input, this->input,
        this->model->num_neurons * sizeof(float),
        cudaMemcpyHostToDevice);
    cudaCheckError("Failed to set input!");
#endif
}

void State::clear_input(int layer_id) {
    int size = this->model->layers[layer_id]->size;
    int offset = this->model->layers[layer_id]->index;

    for (int nid = 0 ; nid < size; ++nid) {
        this->input[nid+offset] = 0.0;
    }
#ifdef PARALLEL
    // Send to GPU
    this->set_input(layer_id, &this->input[offset]);
#endif
}

void* allocate_host(int count, int size) {
    void* ptr = calloc(count, size);
    if (ptr == NULL)
        throw "Failed to allocate space on host for neuron state!";
    return ptr;
}

#ifdef PARALLEL
void* allocate_device(int count, int size, void* source_data) {
    void* ptr;
    cudaMalloc(&ptr, count * size);
    cudaCheckError("Failed to allocate memory on device for neuron state!");
    if (source_data != NULL)
        cudaMemcpy(ptr, source_data, count * size, cudaMemcpyHostToDevice);
    cudaCheckError("Failed to initialize memory on device for neuron state!");
    return ptr;
}
#endif

/* Initializes matrix */
void initialize_matrix(Connection* conn,
        float* mData, int depth) {
    // Multiply by depth if plastic
    int matrix_size = conn->num_weights;
    if (conn->plastic) matrix_size *= depth;

    // Copy data over to a target matrix
    // Parallel requires a temporary matrix be created and copied
    // Serial accesses mData directly
#ifdef PARALLEL
    float *target_matrix = (float*)calloc(matrix_size, sizeof(float));
    if (target_matrix == NULL)
        throw "Failed to allocate temporary matrix on host for randomization!";
#else
    float *target_matrix = mData;
#endif

    // If parameter is specified, interpret it for initialization
    // Otherwise, perform randomization
    if (conn->params.size() > 0) {
        std::stringstream stream(conn->params);

        // Extract first value
        float value;
        stream >> value;

        // If there are no more values, set all weights to first value
        // Otherwise, read values and initialize
        if (stream.eof()) {
            for (int index = 0 ; index < conn->num_weights ; ++index)
                target_matrix[index] = value;
        } else {
            // If parallel, transpose the input...
            // Parallel convergent matrices are laid out such that each
            //   kernel is in a column
#ifdef PARALLEL
            int rows, cols;
            switch (conn->type) {
                case (FULLY_CONNECTED):
                    throw "Cannot specify all weights for fully connected matrix!";
                case (CONVERGENT_CONVOLUTIONAL):
                case (DIVERGENT_CONVOLUTIONAL):
                    rows = conn->overlap * conn->overlap;
                    cols = 1;
                    break;
                case (DIVERGENT):
                    rows = conn->overlap * conn->overlap;
                    cols = conn->from_size;
                case (CONVERGENT):
                    rows = conn->overlap * conn->overlap;
                    cols = conn->to_size;
                    break;
                case (ONE_TO_ONE):
                    rows = conn->from_size;
                    cols = conn->to_size;
                    break;
            }
            for (int col = 0 ; col < cols ; ++col) {
                for (int row = 0 ; row < rows ; ++row) {
                    target_matrix[row * cols + col] = value;
                    if (row != rows-1 and col != cols-1 and stream.eof())
                        throw "Insufficient number of weights specified!";
                    stream >> value;
                }
            }
#else
            for (int index = 0 ; index < conn->num_weights ; ++index) {
                target_matrix[index] = value;
                if (index != conn->num_weights-1 and stream.eof())
                    throw "Insufficient number of weights specified!";
                stream >> value;
            }
#endif
        }
    } else {
        for (int index = 0 ; index < conn->num_weights ; ++index)
            target_matrix[index] = fRand(0, conn->max_weight);
    }

    // Set up further layers if necessary (initialize to zero)
    for (int index = conn->num_weights ; index < matrix_size ; ++index) {
        target_matrix[index] = 0.0;
    }

#ifdef PARALLEL
    cudaMemcpy(mData, target_matrix,
        matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError("Failed to initialize weight matrix!");
    free(target_matrix);
#endif
}

float** build_weight_matrices(Model* model, int depth) {
    // Allocate one big glob for weights
    // Skip shared weights because they don't take up extra space
    int total_size = 0;
    for (int i = 0 ; i < model->num_connections ; ++i) {
        int matrix_size = model->connections[i]->num_weights;
        // If plastic, multiply by depth to make room.
        if (model->connections[i]->plastic)
            matrix_size *= depth;

        if (model->connections[i]->parent == -1)
            total_size += matrix_size;
    }

    float* matrix_datas;
#ifdef PARALLEL
    cudaMalloc((&matrix_datas), total_size * sizeof(float));
    cudaCheckError("Failed to allocate space for weight matrices!");
#else
    matrix_datas = (float*)malloc(total_size * sizeof(float));
    if (matrix_datas == NULL)
        throw "Failed to allocate space for weight matrices!";
#endif

    // Allocate double pointer for indexing purposes
    float** entry_points = 
        (float**)malloc(model->num_connections * sizeof(float*));
    float* curr_point = matrix_datas;
    for (int i = 0 ; i < model->num_connections ; ++i) {
        Connection *conn = model->connections[i];
        if (conn->parent == -1) {
            entry_points[i] = curr_point;
            // Skip over appropriate amount of memory
            // Plastic matrices might have additional layers
            if (conn->plastic)
                curr_point += (conn->num_weights * depth);
            else
                curr_point += conn->num_weights;
        } else {
            entry_points[i] = entry_points[conn->parent];
        }
    }

    // Initialize weights
    for (int i = 0 ; i < model->num_connections ; ++i) {
        Connection *conn = model->connections[i];

        // Skip shared connections
        if (conn->parent = -1)
            initialize_matrix(conn, entry_points[i], depth);
    }
    return entry_points;
}
