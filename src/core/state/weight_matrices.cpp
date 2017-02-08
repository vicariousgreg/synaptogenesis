#include <sstream>

#include "state/state.h"
#include "util/tools.h"
#include "util/error_manager.h"
#include "util/parallel.h"

/* Clears an array */
static void clear(float* vals, int num_vals) {
    for (int i = 0 ; i < num_vals ; ++i) vals[i] = 0.0;
}

/* Randomizes an array */
static void randomize(float* vals, int num_vals, float max) {
    for (int i = 0 ; i < num_vals ; ++i) vals[i] = fRand(0, max);
}

/* Transfers the values from one array to another */
static void transfer(float* from, float* to, int num_vals) {
    for (int i = 0 ; i < num_vals ; ++i) to[i] = from[i];
}

/* Initializes weights using connection parameters */
static void initialize(float* target_matrix, Connection* conn) {
    std::stringstream stream(conn->get_init_params());
    int num_weights = conn->get_num_weights();

    // Extract first value
    float value;
    stream >> value;

    // If there are no more values, set all weights to first value
    // Otherwise, read values and initialize
    if (stream.eof()) {
        for (int index = 0 ; index < num_weights ; ++index)
            target_matrix[index] = value;
    } else {
        // If parallel, transpose the input...
        // Parallel convergent matrices are laid out such that each
        //   kernel is in a column
#ifdef PARALLEL
        int rows, cols;
        int field_size = conn->get_field_size();
        switch (conn->type) {
            case FULLY_CONNECTED:
                ErrorManager::get_instance()->log_error(
                    "Cannot specify all weights for fully connected matrix!");
            case CONVOLUTIONAL:
                rows = field_size * field_size;
                cols = 1;
                break;
            case CONVERGENT:
                rows = field_size * field_size;
                cols = conn->to_layer->size;
                break;
            case ONE_TO_ONE:
                rows = conn->from_layer->size;
                cols = conn->to_layer->size;
                break;
        }
        for (int col = 0 ; col < cols ; ++col) {
            for (int row = 0 ; row < rows ; ++row) {
                target_matrix[row * cols + col] = value;
                if (row != rows-1 and col != cols-1 and stream.eof())
                    ErrorManager::get_instance()->log_error(
                        "Insufficient number of weights specified!");
                stream >> value;
            }
        }
#else
        for (int index = 0 ; index < num_weights ; ++index) {
            target_matrix[index] = value;
            if (index != num_weights-1 and stream.eof())
                ErrorManager::get_instance()->log_error(
                    "Insufficient number of weights specified!");
            stream >> value;
        }
#endif
    }
}

WeightMatrix::WeightMatrix(Connection* conn, int matrix_depth) {
    int num_weights = conn->get_num_weights();
    int matrix_size = num_weights;
    // Multiply by depth if plastic
    if (conn->plastic) matrix_size *= matrix_depth;

    // Copy data over to a target matrix
    // Parallel requires a temporary matrix be created and copied
    // Serial accesses mData directly
#ifdef PARALLEL
    cudaMalloc((&mData), matrix_size * sizeof(float));
    cudaCheckError("Failed to allocate space for weight matrices!");

    float *target_matrix = (float*)calloc(matrix_size, sizeof(float));
    if (target_matrix == NULL)
        ErrorManager::get_instance()->log_error(
            "Failed to allocate temporary matrix on host for randomization!");
#else
    mData = (float*)malloc(matrix_size * sizeof(float));
    if (mData == NULL)
        ErrorManager::get_instance()->log_error(
            "Failed to allocate space for weight matrices!");
    float *target_matrix = mData;
#endif

    // If parameter is specified, interpret it for initialization
    // Otherwise, perform randomization
    if (conn->get_init_params().size() == 0) {
        randomize(target_matrix, num_weights, conn->max_weight);
    } else {
        initialize(target_matrix, conn);
    }

    if (conn->plastic) {
        float *baseline = target_matrix + num_weights;
        transfer(target_matrix, baseline, num_weights);

        float *trace = target_matrix + 2*num_weights;
        clear(trace, num_weights);
    }

#ifdef PARALLEL
    cudaMemcpy(mData, target_matrix,
        matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError("Failed to initialize weight matrix!");
    free(target_matrix);
#endif
}

WeightMatrix::~WeightMatrix() {
#ifdef PARALLEL
    cudaFree(this->mData);
#else
    free(this->mData);
#endif
}

WeightMatrices::WeightMatrices(Model *model, int matrix_depth) {
    // Set up non-shared weights first
    for (auto & conn : model->get_connections())
        if (conn->get_parent() == NULL)
            this->matrices[conn] = new WeightMatrix(conn, matrix_depth);

    // Set up shared weights (just duplicate)
    for (auto & conn : model->get_connections())
        if (conn->get_parent() != NULL)
            this->matrices[conn] = this->matrices[conn->get_parent()];
}

WeightMatrices::~WeightMatrices() {
    for (auto matrix : this->matrices) delete matrix.second;
}
