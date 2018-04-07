#include <algorithm>
#include <sstream>

#include "state/weight_matrix.h"
#include "network/layer.h"
#include "network/connection.h"
#include "engine/kernel/kernel.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/callback_manager.h"
#include "util/logger.h"
#include "util/parallel.h"

/* Sets all values in an array to the given val */
void set_weights(float* arr, int size, float val, float fraction) {
    fSet(arr, size, val, fraction);
}

/* Clears an array */
void clear_weights(float* arr, int size) {
    set_weights(arr, size, 0.0);
}

/* Randomizes an array */
void randomize_weights(float* arr, int size,
        float min, float max, float fraction) {
    fRand(arr, size, min, max, fraction);
}
void randomize_weights_gaussian(float* arr, int size,
        float mean, float std_dev, float max, float fraction) {
    // If standard deviation is 0.0, just set the weights to the mean
    if (std_dev == 0.0) {
        set_weights(arr, size, mean, fraction);
    } else {
        std::normal_distribution<double> dist(mean, std_dev);

        if (fraction == 1.0) {
            for (int i = 0 ; i < size ; ++i)
                arr[i] = std::min((double)max, std::max(0.0, dist(generator)));
        } else {
            std::uniform_real_distribution<double> f_dist(0.0, 1.0);
            for (int i = 0 ; i < size ; ++i)
                arr[i] = (f_dist(generator) < fraction)
                    ? std::min((double)max, std::max(0.0, dist(generator)))
                    : 0.0;
        }
    }
}
void randomize_weights_lognormal(float* arr, int size,
        float mean, float std_dev, float max, float fraction) {
    // If standard deviation is 0.0, just set the weights to the mean
    if (std_dev == 0.0) {
        set_weights(arr, size, mean, fraction);
    } else {
        std::lognormal_distribution<double> dist(mean, std_dev);

        if (fraction == 1.0) {
            for (int i = 0 ; i < size ; ++i)
                arr[i] = std::min((double)max, std::max(0.0, dist(generator)));
        } else {
            std::uniform_real_distribution<double> f_dist(0.0, 1.0);
            for (int i = 0 ; i < size ; ++i)
                arr[i] = (f_dist(generator) < fraction)
                    ? std::min((double)max, std::max(0.0, dist(generator)))
                    : 0.0;
        }
    }
}
void randomize_weights_powerlaw(float* arr, int size,
        float exponent, float min, float max, float fraction) {
    std::uniform_real_distribution<float> dist(0.0, 1.0);

    float coeff_a = pow(max, 1.0-exponent);
    float coeff_b = pow(std::max(min, 0.00001f), 1.0-exponent);
    float coeff = coeff_a - coeff_b;
    float pow_exp = 1.0 / (1.0-exponent);

    if (fraction == 1.0) {
        for (int i = 0 ; i < size ; ++i)
            arr[i] = pow(coeff * dist(generator) + coeff_b, pow_exp);
    } else {
        for (int i = 0 ; i < size ; ++i)
            arr[i] = (dist(generator) < fraction)
                ? pow(coeff * dist(generator) + coeff_b, pow_exp)
                : 0.0;
    }
}

/* Transfers the values from one array to another */
void transfer_weights(float* from, float* to, int size) {
    for (int i = 0 ; i < size ; ++i) to[i] = from[i];
}

/* Clears the diagonal of a weight matrix */
void clear_diagonal_square(float *arr, int rows, int cols) {
    if (rows != cols)
        LOG_ERROR(
            "Attempted to clear diagonal of non-square weight matrix!");

    for (int i = 0 ; i < rows ; ++i)
        arr[i * rows + i] = 0.0;
}

static void initialize_weights(WeightMatrix *matrix,
    const PropertyConfig config, float* target_matrix, Connection* conn);

WeightMatrix::WeightMatrix(Connection* conn)
    : connection(conn),
      device_id(ResourceManager::get_instance()->get_host_id()),
      sparse(false),
      transposed(false),
      pointer(this),
      num_weights(conn->get_num_weights()),
      transpose_flag(false),
      weights(Pointer<float>(conn->get_num_weights())),
      second_order_weights(
          (conn->second_order_host)
              ? Pointer<float>(conn->get_num_weights())
              : Pointer<float>()) {
    this->rows = (conn->convolutional) ? 1 : conn->to_layer->size;
    this->columns = num_weights / rows;
}

WeightMatrix::~WeightMatrix() {
    this->weights.free();
    this->weights_transposed.free();
    this->second_order_weights.free();
    this->nonzero_counts.free();
    this->from_row_indices.free();
    this->from_column_indices.free();
    this->to_row_indices.free();
    this->to_column_indices.free();
    this->used.free();
    this->distances.free();
    this->delays.free();
    for (auto pair : variables) pair.second->free();

#ifdef __CUDACC__
    if (this != this->pointer
            and not ResourceManager::get_instance()->is_host(device_id)) {
        cudaSetDevice(device_id);
        cudaFree(this->pointer);
        ResourceManager::get_instance()->drop_pointer(this->pointer, device_id);
    }
#endif
}

void WeightMatrix::transpose() {
    DeviceID host_id = ResourceManager::get_instance()->get_host_id();

    // Only transpose if necessary
    // If convolutional or num_weights == to_layer size, transposition is a no-op.
    if (not connection->convolutional
        and num_weights != connection->to_layer->size) {

        // Create set of pointers to transpose
        std::vector<BasePointer*> pointers = {
            &weights,
            &weights_transposed,
            &second_order_weights,
            &from_row_indices,
            &from_column_indices,
            &to_row_indices,
            &to_column_indices,
            &used,
            &distances,
            &delays,
        };
        for (auto pair : variables)
            pointers.push_back(pair.second);

        // Transpose on the current device
        if (device_id == host_id) {
            for (auto ptr : pointers)
                if (ptr->get_size() > 0 and ptr->get_device_id() == device_id)
                    transpose_matrix_in_place<float>(
                        (float*)ptr->get(), get_rows(), get_columns());
        } else {
#ifdef __CUDACC__
            // Create temporary matrix
            Pointer<float> temp =
                Pointer<float>::device_pointer(device_id, num_weights);

            dim3 dimGrid = calc_transpose_blocks(get_rows(), get_columns());
            dim3 dimBlock = calc_transpose_threads(get_rows(), get_columns());

            auto stream =
                ResourceManager::get_instance()->get_default_stream(device_id);

            // Copy to temp, transpose back into original memory (not in place)
            for (auto ptr : pointers) {
                if (ptr->get_size() > 0 and ptr->get_device_id() == device_id) {
                    Pointer<float> p = Pointer<float>(
                        (float*)ptr->get(), num_weights, device_id, false);
                    p.copy_to(&temp, stream);
                    cudaSetDevice(device_id);
                    transpose_matrix_parallel<float>
                        <<<dimGrid, dimBlock, 0, stream->get_cuda_stream()>>>
                        (temp, p, get_rows(), get_columns());

                }
            }

            device_synchronize();
            device_check_error("Failed to transpose weight matrix!");
            temp.free();
#endif
        }
    }

    this->transposed = not this->transposed;
}

void WeightMatrix::set_transpose_flag(bool t) {
    if (transpose_flag and (not t)) {
        weights_transposed.free();
        weights_transposed = Pointer<float>();
    } else if ((not transpose_flag) and t) {
        weights_transposed = Pointer<float>(num_weights);
    }
    transpose_flag = t;
}

void WeightMatrix::resize() {
    if (weights.get_size() != num_weights) {
        this->sparse = true;
        this->num_weights = weights.get_size();
        this->connection->sparsify(this->num_weights);
        this->rows = connection->to_layer->size;
        this->columns = num_weights / rows;
    }
}

/* This function is necessary for sparsified wrapping arborized connections
 * Indices cannot be remapped right away because they will corrupt
 *   delay initialization */
void WeightMatrix::adjust_sparse_indices() {
    if (sparse) {
        int max_nonzero = num_weights / connection->to_layer->size;
        int from_rows = connection->from_layer->rows;
        int from_columns = connection->from_layer->columns;
        for (int row = 0 ; row < rows ; ++row) {
            int curr_col = 0;
            for (int col = 0 ; col < columns ; ++col) {
                int new_index = row*max_nonzero + curr_col++;
                if (weights[new_index] != 0.0) {
                    int from_row = from_row_indices[new_index];
                    int from_column = from_column_indices[new_index];

                    from_row = (from_row < 0)
                        ? from_row + from_rows
                        : (from_row >= from_rows)
                            ? from_row - from_rows : from_row;
                    from_column = (from_column < 0)
                        ? from_column + from_columns
                        : (from_column >= from_columns)
                            ? from_column - from_columns : from_column;

                    from_row_indices[new_index] = from_row;
                    from_column_indices[new_index] = from_column;
                }
            }
        }
    }
}


void WeightMatrix::transfer(DeviceID new_device) {
#ifdef __CUDACC__
    if (device_id == new_device) return;

    auto host_id = ResourceManager::get_instance()->get_host_id();
    if (device_id != host_id and new_device != host_id)
        LOG_ERROR("Cannot transfer attributes directly between devices!");

    if (new_device == host_id) {
        auto old_device = device_id;
        auto old_ptr = this->pointer;

        // Transfer to host
        cudaMemcpy(this, this->pointer,
            get_object_size(), cudaMemcpyDeviceToHost);
        this->device_id = new_device;

        // Free old device copy
        cudaSetDevice(old_device);
        cudaFree(old_ptr);
        ResourceManager::get_instance()->drop_pointer(old_ptr, old_device);
        this->pointer = this;
    } else {
        // Transfer to device
        this->device_id = new_device;
        cudaSetDevice(new_device);
        this->pointer = (WeightMatrix*)
            ResourceManager::get_instance()->allocate_device(
                1, get_object_size(), this, new_device);
    }
    device_synchronize();
    device_check_error("Failed to transfer WeightMatrix to device!");
#endif
}

BasePointer* WeightMatrix::get_layer(std::string key) {
    if (key == "weights") return &weights;
    try {
        return variables.at(key);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Failed to retrieve data \"" + key + "\" in WeightMatrix for "
            "connection:" + connection->str());
    }
}

std::vector<BasePointer*> WeightMatrix::get_pointers() {
    std::vector<BasePointer*> pointers;
    for (auto pair : get_pointer_map())
        pointers.push_back(pair.second);
    return pointers;
}

std::map<PointerKey, BasePointer*> WeightMatrix::get_pointer_map() {
    std::map<PointerKey, BasePointer*> pointers;

    pointers[PointerKey(connection->id, "weights",
        weights.get_bytes())] = &weights;
    pointers[PointerKey(connection->id, "weights transposed",
        weights_transposed.get_bytes())] = &weights_transposed;
    pointers[PointerKey(connection->id, "second order weights",
        second_order_weights.get_bytes())] = &second_order_weights;
    pointers[PointerKey(connection->id, "nonzero counts",
        nonzero_counts.get_bytes())] = &nonzero_counts;
    pointers[PointerKey(connection->id, "from row indices",
        from_row_indices.get_bytes())] = &from_row_indices;
    pointers[PointerKey(connection->id, "from column indices",
        from_column_indices.get_bytes())] = &from_column_indices;
    pointers[PointerKey(connection->id, "to row indices",
        to_row_indices.get_bytes())] = &to_row_indices;
    pointers[PointerKey(connection->id, "to column indices",
        to_column_indices.get_bytes())] = &to_column_indices;
    pointers[PointerKey(connection->id, "used",
        used.get_bytes())] = &used;
    pointers[PointerKey(connection->id, "distances",
        distances.get_bytes())] = &distances;
    pointers[PointerKey(connection->id, "delays",
        delays.get_bytes())] = &delays;

    for (auto pair : variables)
        pointers[PointerKey(
            connection->id, pair.first,
            pair.second->get_bytes())] = pair.second;
    return pointers;
}

WeightMatrix *WeightMatrix::build(Connection *conn) {
    auto mat = new WeightMatrix(conn);
    mat->init();
    return mat;
}

void WeightMatrix::init() {
    initialize_weights(this,
        connection->get_config()->get_weight_config(), weights, connection);
    if (connection->sparse) sparsify();
    register_variables();
}

void WeightMatrix::sparsify() {
    if (connection->convolutional)
        LOG_ERROR("Error intializing weight matrix for " + connection->str() +
        "\n   Cannot sparsify convolutional weight matrix!");

    // Get the full indices
    this->get_indices();

    // Compute nonzero weight counts and sparse matrix size
    int max_nonzero = 0;
    for (int row = 0 ; row < rows ; ++row) {
        int nonzero = 0;
        for (int col = 0 ; col < columns ; ++col) {
            int index = row*columns + col;
            if (weights[index] == 0.0) used[index] = 0;
            nonzero += used[index];
        }
        max_nonzero = MAX(max_nonzero, nonzero);
    }
    int sparse_num_weights = max_nonzero * connection->to_layer->size;

    // Ensure nonzero size
    if (sparse_num_weights == 0) {
        LOG_WARNING(
            "Warning in weight config for " + connection->str() + ":\n" +
            "    Attempted to sparsify empty matrix!");
        sparse_num_weights = connection->to_layer->size;
    } else if (sparse_num_weights == num_weights) return;

    // Create index matrices (padded with -1) and nonzero counts
    auto compact_from_row_indices = Pointer<int>(sparse_num_weights, -1);
    auto compact_from_column_indices = Pointer<int>(sparse_num_weights, -1);
    auto compact_to_row_indices = Pointer<int>(sparse_num_weights, -1);
    auto compact_to_column_indices = Pointer<int>(sparse_num_weights, -1);
    auto compact_used = Pointer<int>(sparse_num_weights, 0);
    this->nonzero_counts = Pointer<int>(connection->to_layer->size, 0);

    // Create new weight matrix
    // Weights_transposed can't be created yet
    // Second order connections cannot be sparse
    auto new_weights = Pointer<float>(sparse_num_weights, 0.0);

    // Condense
    int total_nonzero = 0;
    for (int row = 0 ; row < rows ; ++row) {
        int curr_col = 0;
        for (int col = 0 ; col < columns ; ++col) {
            int old_index = row*columns + col;

            if (weights[old_index] != 0.0 and used[old_index]) {
                int new_index = row*max_nonzero + curr_col++;

                new_weights[new_index] = weights[old_index];
                compact_from_row_indices[new_index]
                    = from_row_indices[old_index];
                compact_from_column_indices[new_index]
                    = from_column_indices[old_index];
                compact_to_row_indices[new_index]
                    = to_row_indices[old_index];
                compact_to_column_indices[new_index]
                    = to_column_indices[old_index];
                compact_used[new_index]
                    = used[old_index];
            }
        }
        nonzero_counts[row] = curr_col;
        total_nonzero += curr_col;
        while (curr_col < max_nonzero)
            new_weights[row*max_nonzero + curr_col++] = 0.0;
    }

    // Print compression statistics
    printf("Sparsified %s:\n"
           "  Compression:         %12d / %12d (%8.6f)\n"
           "  Theoretical minimum: %12d / %12d (%8.6f)\n",
        connection->str().c_str(),
        sparse_num_weights, num_weights,
        float(sparse_num_weights) / num_weights,
        total_nonzero, num_weights,
        float(total_nonzero) / num_weights);

    // Free intermediate data and move pointers
    this->from_row_indices.free();
    this->from_column_indices.free();
    this->to_row_indices.free();
    this->to_column_indices.free();
    this->used.free();
    this->from_row_indices = Pointer<int>(compact_from_row_indices, true);
    this->from_column_indices = Pointer<int>(compact_from_column_indices, true);
    this->to_row_indices = Pointer<int>(compact_to_row_indices, true);
    this->to_column_indices = Pointer<int>(compact_to_column_indices, true);
    this->used = Pointer<int>(compact_used, true);

    // Free old weights and move pointer
    this->weights.free();
    this->weights = Pointer<float>(new_weights, true); // claim_ownership = true

    // Resize, updating necessary variables
    this->resize();
}

template Pointer<float> WeightMatrix::create_variable();
template Pointer<int> WeightMatrix::create_variable();
template <class T>
Pointer<T> WeightMatrix::create_variable() {
    return Pointer<T>(num_weights);
}

void WeightMatrix::register_variable(
        std::string key, BasePointer* ptr) {
    if (this->variables.count(key) > 0)
        LOG_ERROR(
            "Repeated weight matrix variable key: " + key);
    this->variables[key] = ptr;
}

/******************************************************************************/
/**************************** WEIGHT INITIALIZATION ***************************/
/******************************************************************************/
static void flat_config(const PropertyConfig& config, float* target_matrix,
        Connection* conn) {
    float weight = config.get_float("weight", 1.0);
    float fraction = config.get_float("fraction", 1.0);

    set_weights(target_matrix, conn->get_num_weights(), weight, fraction);
}

static void random_config(const PropertyConfig& config, float* target_matrix,
        Connection* conn) {
    float max_weight = config.get_float("max weight", conn->max_weight);
    float min_weight = config.get_float("min weight", 0.0);
    float fraction = config.get_float("fraction", 1.0);

    randomize_weights(target_matrix, conn->get_num_weights(),
        min_weight, max_weight, fraction);
}

static void gaussian_config(const PropertyConfig& config, float* target_matrix,
        Connection* conn) {
    float mean = config.get_float("mean", 1.0);
    float std_dev = config.get_float("std dev", 0.3);
    float fraction = config.get_float("fraction", 1.0);

    if (std_dev < 0)
        LOG_ERROR(
            "Error in weight config for " + conn->str() + ":\n"
            "  Gaussian weight config std_dev must be positive!");

    randomize_weights_gaussian(target_matrix, conn->get_num_weights(),
        mean, std_dev, conn->max_weight, fraction);
}

static void log_normal_config(const PropertyConfig& config,
        float* target_matrix, Connection* conn) {
    float mean = config.get_float("mean", 1.0);
    float std_dev = config.get_float("std dev", 0.3);
    float fraction = config.get_float("fraction", 1.0);

    if (std_dev < 0)
        LOG_ERROR(
            "Error in weight config for " + conn->str() + ":\n"
            "  Log normal weight config std_dev must be positive!");

    randomize_weights_lognormal(target_matrix, conn->get_num_weights(),
        mean, std_dev, conn->max_weight, fraction);
}

static void power_law_config(const PropertyConfig& config, float* target_matrix,
        Connection* conn) {
    float exponent = config.get_float("exponent", 1.5);
    float fraction = config.get_float("fraction", 1.0);
    float max_weight = config.get_float("max weight", conn->max_weight);
    float min_weight = config.get_float("min weight", 0.0);

    exponent = abs(exponent);

    if (conn->max_weight < 0)
        LOG_ERROR(
            "Error in weight config for " + conn->str() + ":\n"
            "  Power law config max must be positive!");

    randomize_weights_powerlaw(target_matrix, conn->get_num_weights(),
        exponent, min_weight, max_weight, fraction);
}

static void circular_mask_config(const PropertyConfig& config,
        float* target_matrix, Connection* conn) {
    if (conn->get_type() != CONVERGENT)
        LOG_ERROR(
            "Error in weight config for " + conn->str() + ":\n"
            "  CircularMaskConfig can only be used "
            "on convergent arborized connections!");

    auto ac = conn->get_config()->get_arborized_config();
    int row_field_size = ac.column_field_size;
    int col_field_size = ac.column_field_size;
    int kernel_size = ac.get_total_field_size();

    float row_radius = config.get_float("row radius", 0);
    float col_radius = config.get_float("column radius", 0);
    float row_diameter = config.get_float("row diameter", 0);
    float col_diameter = config.get_float("column diameter", 0);
    float radius = config.get_float("radius", 0);
    float diameter = config.get_float("diameter", 0);
    bool invert = config.get_bool("invert", false);
    float value = config.get_float("value", 0.0);

    if (row_radius <= 0) {
        if (row_diameter > 0)
            row_radius = row_diameter / 2;
        else if (radius > 0)
            row_radius = radius;
        else if (diameter > 0)
            row_radius = diameter / 2;
        else
            row_radius = float(row_field_size) / 2;
    }

    if (col_radius <= 0) {
        if (col_diameter > 0)
            col_radius = col_diameter / 2;
        else if (radius > 0)
            col_radius = radius;
        else if (diameter > 0)
            col_radius = diameter / 2;
        else
            col_radius = float(col_field_size) / 2;
    }

    if (row_radius <= 0 or col_radius <= 0)
        LOG_ERROR(
            "Error in weight config for " + conn->str() + ":\n"
            "  Circular mask weight config radii must be positive!");

    float row_center = float(row_field_size) / 2;
    float col_center = float(col_field_size) / 2;
    float row_radius_sq = pow(row_radius, 2);
    float col_radius_sq = pow(col_radius, 2);
    float sq_mult = row_radius_sq * col_radius_sq;

    // Convolutional connections are unique in that there is only one kernel.
    int size = (conn->convolutional) ? 1 : conn->to_layer->size;

    for (int index = 0 ; index < size ; ++index) {
        int weight_offset = index * kernel_size;

        for (int k_row = 0 ; k_row < row_field_size ; ++k_row) {
            for (int k_col = 0 ; k_col < col_field_size ; ++k_col) {
                float term =
                    (powf(k_row + 0.5 - row_center, 2) * col_radius_sq) +
                    (powf(k_col + 0.5 - col_center, 2) * row_radius_sq);

                if (invert == (term <= sq_mult)) {
                    int weight_index = weight_offset +
                        (k_row * col_field_size) + k_col;
                    target_matrix[weight_index] = value;
                }
            }
        }
    }
}

static void specified_config(const PropertyConfig& config, float* target_matrix,
        Connection* conn) {
    std::string weight_string = config.get("weight string", "");
    if (weight_string == "")
        LOG_ERROR(
            "Error in weight config for " + conn->str() + ":\n"
            "  Missing weight string for specified weight config!");

    std::stringstream stream(weight_string);
    int num_weights = conn->get_num_weights();

    int rows = conn->to_layer->size;
    int cols = conn->from_layer->size;
    auto ac = conn->get_config()->get_arborized_config();
    int row_field_size = ac.row_field_size;
    int column_field_size = ac.column_field_size;
    switch (conn->get_type()) {
        case CONVERGENT:
            if (conn->convolutional) {
                rows = 1;
                cols = row_field_size * column_field_size;
            } else {
                rows = conn->to_layer->size;
                cols = row_field_size * column_field_size;
            }
            break;
        case DIVERGENT:
            if (conn->convolutional) {
                rows = 1;
                cols = row_field_size * column_field_size;
            } else {
                rows = conn->from_layer->size;
                cols = row_field_size * column_field_size;
            }
            break;
        case ONE_TO_ONE:
            rows = 1;
            break;
    }

    float value;
    for (int row = 0 ; row < rows ; ++row) {
        for (int col = 0 ; col < cols ; ++col) {
            if (row != rows-1 and col != cols-1 and stream.eof())
                LOG_ERROR(
                    "Error in weight config for " + conn->str() + ":\n"
                    "  Insufficient number of weights specified!");
            else stream >> value;
            target_matrix[row * cols + col] = value;
        }
    }
}

static void weight_callback_config(WeightMatrix *matrix,
        const PropertyConfig& config, float* target_matrix, Connection* conn) {
    if (not config.has("callback"))
        LOG_ERROR(
            "Unspecified weight callback function for connection "
            + conn->str());

    void (*callback)(int, int, void*) =
        CallbackManager::get_instance()->get_weight_callback(
            config.get("callback"));

    int id = config.get_int("id", 0);
    int num_weights = conn->get_num_weights();
    callback(id, num_weights, target_matrix);
}

static void distance_weight_callback_config(WeightMatrix *matrix,
        const PropertyConfig& config, float* target_matrix, Connection* conn) {
    if (not config.has("distance callback"))
        LOG_ERROR(
            "Unspecified distance callback function for connection "
            + conn->str());

    void (*callback)(int, int, void*, void*) =
        CallbackManager::get_instance()->get_distance_weight_callback(
            config.get("distance callback"));

    int id = config.get_int("id", 0);
    float from_spacing = config.get_float("from spacing", 1.0);
    float to_spacing = config.get_float("to spacing", 1.0);
    float x_offset = config.get_float("x offset", 0.0);
    float y_offset = config.get_float("y offset", 0.0);

    int num_weights = conn->get_num_weights();
    matrix->get_distances(from_spacing, to_spacing, x_offset, y_offset);

    callback(id, num_weights, target_matrix, matrix->distances.get());
}

static void clear_diagonal(WeightMatrix *matrix,
        const PropertyConfig& config, float* target_matrix, Connection* conn) {
    switch(conn->get_type()) {
        case FULLY_CONNECTED:
            clear_diagonal_square(target_matrix,
                conn->from_layer->size, conn->to_layer->size);
            break;
        case SUBSET: {
            auto subset_config = conn->get_config()->get_subset_config();
            clear_diagonal_square(target_matrix,
                subset_config.from_size,
                subset_config.to_size);
            break;
        }
        case CONVERGENT:
            // Use inverted circular mask config with radius 0.5
            circular_mask_config(
                PropertyConfig({{"radius", "0.5"}, {"invert", "true"}}),
                    target_matrix, conn);
            break;
    }
}

static void initialize_weights(WeightMatrix *matrix,
        const PropertyConfig config, float* target_matrix, Connection* conn) {
    if (config.has("fraction")) {
        float fraction = config.get_float("fraction", 1.0);
        if (fraction < 0 or fraction > 1.0)
            LOG_ERROR(
                "Error in weight config for " + conn->str() + ":\n"
                "  Weight config fraction must be between 0 and 1!");
    }

    auto type = config.get("type", "flat");

    if (type == "flat")
        flat_config(config, target_matrix, conn);
    else if (type == "random")
        random_config(config, target_matrix, conn);
    else if (type == "gaussian")
        gaussian_config(config, target_matrix, conn);
    else if (type == "log normal")
        log_normal_config(config, target_matrix, conn);
    else if (type == "power law")
        power_law_config(config, target_matrix, conn);
    else if (type == "specified")
        specified_config(config, target_matrix, conn);
    else
        LOG_ERROR(
            "Error in weight config for " + conn->str() + ":\n"
            "  Unrecognized weight config type: " + type);

    // Now, run callbacks
    if (config.has("callback"))
        weight_callback_config(matrix, config, target_matrix, conn);

    if (config.has("distance callback"))
        distance_weight_callback_config(matrix, config, target_matrix, conn);

    // Finally, do mask processing
    if (config.has_child("circular mask"))
        circular_mask_config(config.get_child("circular mask"),
            target_matrix, conn);

    if (config.has_child_array("circular mask"))
        for (auto child_config : config.get_child_array("circular mask"))
            circular_mask_config(child_config, target_matrix, conn);

    if (not config.get_bool("diagonal", true))
        clear_diagonal(matrix, config, target_matrix, conn);
}

/******************************************************************************/
/************************ PAIRWISE WEIGHT OPERATIONS **************************/
/******************************************************************************/
#define INDICES_EXTRACTIONS \
    const WeightMatrix* mat = synapse_data.matrix; \
    int* from_row_is = mat->from_row_indices.get(); \
    int* from_column_is = mat->from_column_indices.get(); \
    int* to_row_indices = mat->to_row_indices.get(); \
    int* to_column_indices = mat->to_column_indices.get(); \
    int* used = mat->used.get(); \

#define SET_INDICES(PREFIX) \
    from_row_is[weight_index] = PREFIX##from_row; \
    from_column_is[weight_index] = PREFIX##from_column; \
    to_row_indices[weight_index] = to_row; \
    to_column_indices[weight_index] = to_column; \
    used[weight_index] = 1;

CALC_ALL(indices_kernel,
    INDICES_EXTRACTIONS,
    ,
    SET_INDICES(),
)

CALC_CONVERGENT(indices_kernel_wrap_convergent,
    INDICES_EXTRACTIONS,
    ,
    SET_INDICES(pre_wrap_),
)
CALC_DIVERGENT(indices_kernel_wrap_divergent,
    INDICES_EXTRACTIONS,
    ,
    SET_INDICES(pre_wrap_),
)
CALC_CONVERGENT_CONVOLUTIONAL_BY_WEIGHT(indices_kernel_wrap_convergent_convolutional,
    INDICES_EXTRACTIONS,
    ,
    SET_INDICES(pre_wrap_);,
)
CALC_DIVERGENT_CONVOLUTIONAL_BY_WEIGHT(indices_kernel_wrap_divergent_convolutional,
    INDICES_EXTRACTIONS,
    ,
    SET_INDICES(pre_wrap_),
)

void WeightMatrix::get_indices() {
    // If used is already set, indices have already been retrieved
    if (not this->used.is_null()) return;

    auto conn = this->connection;
    int num_weights = conn->get_num_weights();
    LOG_DEBUG("Retrieving to/from indices for : " + conn->str());

    // Retrieve appropriate kernel
    Kernel<SYNAPSE_ARGS> indices_kernel;
    try {
        switch(conn->get_type()) {
            case CONVERGENT:
                indices_kernel = (conn->convolutional)
                    ? get_indices_kernel_wrap_convergent_convolutional()
                    : get_indices_kernel_wrap_convergent();
                break;
            case DIVERGENT:
                indices_kernel = (conn->convolutional)
                    ? get_indices_kernel_wrap_divergent_convolutional()
                    : get_indices_kernel_wrap_divergent();
                break;
            default:
                indices_kernel = indices_kernel_map.at(conn->get_type());
                break;
        }
    } catch (std::out_of_range) {
        LOG_ERROR("Unrecognized connection type!");
    }

    auto res_man = ResourceManager::get_instance();
    this->from_row_indices = Pointer<int>(num_weights, -1);
    this->from_column_indices = Pointer<int>(num_weights, -1);
    this->to_row_indices = Pointer<int>(num_weights, -1);
    this->to_column_indices = Pointer<int>(num_weights, -1);
    this->used = Pointer<int>(num_weights, 0);

    if (res_man->get_gpu_ids().size() == 0) {
        LOG_DEBUG("  Running serial kernel...");
        indices_kernel.run_serial(SynapseData(this, conn));
    } else {
#ifdef __CUDACC__
        DeviceID device_id = *res_man->get_devices().begin();
        DeviceID host_id = res_man->get_host_id();
        Stream *stream = res_man->get_default_stream(device_id);

        bool transferred = false;

        // Transfer to device if necessary
        if (this->get_device_id() == host_id) {
            LOG_DEBUG("  Tranferring WeightMatrix to host...");

            std::vector<BasePointer*> pointers;
            pointers.push_back(&this->from_row_indices);
            pointers.push_back(&this->from_column_indices);
            pointers.push_back(&this->to_row_indices);
            pointers.push_back(&this->to_column_indices);
            pointers.push_back(&this->used);

            for (auto ptr : pointers)
                ptr->transfer(device_id);

            this->transfer(device_id);
            this->transpose();
            transferred = true;
        } else {
            device_id = this->get_device_id();
            stream = res_man->get_default_stream(device_id);
        }

        LOG_DEBUG("  Running parallel kernel...");

        // Run parallel kernel
        dim3 threads = (conn->convolutional)
            ? calc_threads(num_weights)
            : calc_threads(conn->to_layer->size);
        dim3 blocks = (conn->convolutional)
            ? calc_blocks(num_weights)
            : calc_blocks(conn->to_layer->size);
        indices_kernel.run_parallel(
            stream, blocks, threads, SynapseData(this->pointer, conn));
        device_synchronize();
        device_check_error("Failed to compute weight indices!");

        if (transferred) {
            // Transfer back to host
            this->transpose();
            this->transfer(host_id);

            std::vector<BasePointer*> pointers;
            pointers.push_back(&this->from_row_indices);
            pointers.push_back(&this->from_column_indices);
            pointers.push_back(&this->to_row_indices);
            pointers.push_back(&this->to_column_indices);
            pointers.push_back(&this->used);

            for (auto ptr : pointers)
                ptr->transfer(host_id);

            device_synchronize();
            device_check_error("Failed to compute weight indices!");
        }
#endif
    }
}

/******************************* DISTANCES ************************************/

HOST void compute_distances_SERIAL(
        int num_weights, float* distances,
        int* from_row_indices, int* from_column_indices,
        int* to_row_indices, int* to_column_indices,
        float from_spacing, float to_spacing,
        float x_offset, float y_offset) {

    for (int weight_index = 0 ; weight_index < num_weights ; ++weight_index) {
        float f_y = from_row_indices[weight_index] * from_spacing;
        float f_x = from_column_indices[weight_index] * from_spacing;

        float t_y = to_row_indices[weight_index] * to_spacing + y_offset;
        float t_x = to_column_indices[weight_index] * to_spacing + x_offset;

        distances[weight_index] = pow(
            pow(t_x - f_x, 2) + pow(t_y - f_y, 2),
            0.5);
    }
}

#ifdef __CUDACC__
GLOBAL void compute_distances_PARALLEL(
        int num_weights, float* distances,
        int* from_row_indices, int* from_column_indices,
        int* to_row_indices, int* to_column_indices,
        float from_spacing, float to_spacing,
        float x_offset, float y_offset) {

    int weight_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (weight_index < num_weights) {
        float f_y = from_row_indices[weight_index] * from_spacing;
        float f_x = from_column_indices[weight_index] * from_spacing;

        float t_y = to_row_indices[weight_index] * to_spacing + y_offset;
        float t_x = to_column_indices[weight_index] * to_spacing + x_offset;

        distances[weight_index] = pow(
            pow(t_x - f_x, 2) + pow(t_y - f_y, 2),
            0.5);
    }
}
#endif

void WeightMatrix::get_distances(float from_spacing, float to_spacing,
        float x_offset, float y_offset) {
    // If distances is already set, distances have already been retrieved
    if (not this->distances.is_null()) return;

    // Ensure indices have been computed
    if (this->used.is_null())
        this->get_indices();

    auto conn = this->connection;
    LOG_DEBUG("Retrieving distances for : " + conn->str());

    auto res_man = ResourceManager::get_instance();
    int num_weights = conn->get_num_weights();
    this->distances = Pointer<float>(num_weights, 0.0);

    if (res_man->get_gpu_ids().size() == 0) {
        LOG_DEBUG("  Running serial kernel...");
        compute_distances_SERIAL(num_weights,
            this->distances.get(),
            this->from_row_indices.get(),
            this->from_column_indices.get(),
            this->to_row_indices.get(),
            this->to_column_indices.get(),
            from_spacing, to_spacing, x_offset, y_offset);
    } else {
#ifdef __CUDACC__
        DeviceID device_id = *res_man->get_devices().begin();
        DeviceID host_id = res_man->get_host_id();
        Stream *stream = res_man->get_default_stream(device_id);

        // Transfer to device if necessary
        if (this->get_device_id() == host_id) {
            LOG_DEBUG("  Tranferring WeightMatrix to host...");
            this->distances.transfer(device_id);
            this->from_row_indices.transfer(device_id);
            this->from_column_indices.transfer(device_id);
            this->to_row_indices.transfer(device_id);
            this->to_column_indices.transfer(device_id);
        } else {
            device_id = this->get_device_id();
            stream = res_man->get_default_stream(device_id);
        }

        LOG_DEBUG("  Running parallel kernel...");

        // Run parallel kernel
        dim3 threads = calc_threads(num_weights);
        dim3 blocks = calc_blocks(num_weights);
        cudaSetDevice(device_id);

        compute_distances_PARALLEL
        <<<blocks, threads, 0, stream->get_cuda_stream()>>>
            (num_weights,
                this->distances.get_unsafe(),
                this->from_row_indices.get_unsafe(),
                this->from_column_indices.get_unsafe(),
                this->to_row_indices.get_unsafe(),
                this->to_column_indices.get_unsafe(),
                from_spacing, to_spacing,
                x_offset, y_offset);
        device_synchronize();
        device_check_error("Failed to compute distances!");

        // Transfer back to host
        this->distances.transfer(host_id);
        this->from_row_indices.transfer(host_id);
        this->from_column_indices.transfer(host_id);
        this->to_row_indices.transfer(host_id);
        this->to_column_indices.transfer(host_id);

        device_synchronize();
        device_check_error("Failed to compute weight indices!");
#endif
    }
}

/********************************* DELAYS *************************************/

HOST void compute_delays_SERIAL(int num_weights,
        int* delays, float* distances,
        float from_spacing, float to_spacing,
        float x_offset, float y_offset,
        float velocity, bool cap_delay, int base_delay) {

    for (int weight_index = 0 ; weight_index < num_weights ; ++weight_index) {
        int delay = base_delay + (distances[weight_index] / velocity);
        if (delay > 31 and not cap_delay) {
            printf("BIT delays cannot be greater than 31!");
            assert(false);
        }
        delays[weight_index] = MIN(31, delay);
    }
}

#ifdef __CUDACC__
GLOBAL void compute_delays_PARALLEL(int num_weights,
        int* delays, float* distances,
        float from_spacing, float to_spacing,
        float x_offset, float y_offset,
        float velocity, bool cap_delay, int base_delay) {

    int weight_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (weight_index < num_weights) {
        int delay = base_delay + (distances[weight_index] / velocity);
        if (delay > 31 and not cap_delay) {
            printf("BIT delays cannot be greater than 31!");
            assert(false);
        }
        delays[weight_index] = MIN(31, delay);
    }
}
#endif

void WeightMatrix::get_delays(OutputType output_type,
        float from_spacing, float to_spacing,
        float x_offset, float y_offset,
        float velocity, bool cap_delay) {

    // If delays is already set, delays have already been retrieved
    if (not this->delays.is_null()) return;

    // Ensure distances have been computed
    if (this->distances.is_null())
        this->get_distances(from_spacing, to_spacing, x_offset, y_offset);

    auto conn = this->connection;
    LOG_DEBUG("Initializing delays for : " + conn->str());
    if (output_type != BIT)
        LOG_ERROR("Only BIT output connections can have variable delays!");

    int base_delay = conn->delay;
    LOG_DEBUG("Retrieving delays for : " + conn->str());

    auto res_man = ResourceManager::get_instance();
    int num_weights = conn->get_num_weights();
    this->delays = Pointer<int>(num_weights, 0);

    if (res_man->get_gpu_ids().size() == 0) {
        LOG_DEBUG("  Running serial kernel...");
        compute_delays_SERIAL(num_weights,
            this->delays.get(), this->distances.get(),
            from_spacing, to_spacing, x_offset, y_offset,
            velocity, cap_delay, conn->delay);
    } else {
#ifdef __CUDACC__
        DeviceID device_id = *res_man->get_devices().begin();
        DeviceID host_id = res_man->get_host_id();
        Stream *stream = res_man->get_default_stream(device_id);

        // Transfer to device if necessary
        if (this->get_device_id() == host_id) {
            LOG_DEBUG("  Tranferring WeightMatrix to host...");
            this->distances.transfer(device_id);
            this->delays.transfer(device_id);
        } else {
            device_id = this->get_device_id();
            stream = res_man->get_default_stream(device_id);
        }

        LOG_DEBUG("  Running parallel kernel...");

        // Run parallel kernel
        dim3 threads = calc_threads(num_weights);
        dim3 blocks = calc_blocks(num_weights);
        cudaSetDevice(device_id);

        compute_delays_PARALLEL
        <<<blocks, threads, 0, stream->get_cuda_stream()>>>
            (num_weights,
                this->delays.get_unsafe(), this->distances.get_unsafe(),
                from_spacing, to_spacing,
                x_offset, y_offset,
                velocity, cap_delay, conn->delay);
        device_synchronize();
        device_check_error("Failed to compute delays!");

        // Transfer back to host
        this->distances.transfer(host_id);
        this->delays.transfer(host_id);

        device_synchronize();
        device_check_error("Failed to compute weight indices!");
#endif
    }
}

/******************************************************************************/
/**************************** MATRIX TRANSPOSITION ****************************/
/******************************************************************************/

/* Adapted from StackOverflow implementation of "Following the cycles" in-place
 *  transpose algorithm by Christian Ammer:
 * https://stackoverflow.com/questions/9227747/in-place-transposition-of-a-matrix
 */

template<class RandomIterator>
static void transpose_in_place_impl(RandomIterator first, RandomIterator last,
        long desired_rows) {
    const long mn1 = (last - first - 1);
    const long n   = (last - first) / desired_rows;
    std::vector<bool> visited(last - first);
    RandomIterator cycle = first;
    while (++cycle != last) {
        if (visited[cycle - first]) continue;
        long a = cycle - first;
        do {
            a = a == mn1 ? mn1 : (n * a) % mn1;
            std::swap(*(first + a), *cycle);
            visited[a] = true;
        } while ((first + a) != cycle);
    }
}

template void transpose_matrix_in_place<float>(
    float* data, int original_rows, int original_cols);
template void transpose_matrix_in_place<int>(
    int* data, int original_rows, int original_cols);

template <typename T>
void transpose_matrix_in_place(T* data, int original_rows, int original_cols) {
    transpose_in_place_impl(data,
        data + (original_rows * original_cols), original_cols);
}
