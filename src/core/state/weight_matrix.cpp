#include <algorithm>
#include <sstream>

#include "state/weight_matrix.h"
#include "network/layer.h"
#include "network/connection.h"
#include "engine/kernel/kernel.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/callback_manager.h"
#include "util/error_manager.h"
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
    this->nonzero_counts.free();
    this->from_row_indices.free();
    this->from_column_indices.free();
    this->second_order_weights.free();
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
    // If num_weights == to_layer size, transposition is a no-op.
    if (num_weights != connection->to_layer->size) {
        // Transpose on the current device
        if (device_id == host_id) {
            transpose_matrix_in_place<float>(
                this->weights.get(), get_rows(), get_columns());
            if (this->sparse) {
                transpose_matrix_in_place<int>(
                    this->from_row_indices.get(), get_rows(), get_columns());
                transpose_matrix_in_place<int>(
                    this->from_column_indices.get(), get_rows(), get_columns());
            }
            for (auto pair : variables)
                transpose_matrix_in_place<float>(
                    (float*)pair.second->get(), get_rows(), get_columns());
        } else {
#ifdef __CUDACC__
            // Create temporary matrix
            Pointer<float> temp =
                Pointer<float>::device_pointer(device_id, num_weights);

            dim3 dimGrid = calc_transpose_blocks(get_rows(), get_columns());
            dim3 dimBlock = calc_transpose_threads(get_rows(), get_columns());

            auto stream =
                ResourceManager::get_instance()->get_default_stream(device_id);

            this->weights.copy_to(temp, stream);
            cudaSetDevice(device_id);
            transpose_matrix_parallel<float>
                <<<dimGrid, dimBlock, 0, stream->get_cuda_stream()>>>
                (temp, this->weights, get_rows(), get_columns());

            if (this->sparse) {
                this->from_row_indices.cast<float>().copy_to(temp, stream);
                cudaSetDevice(device_id);
                transpose_matrix_parallel<float>
                    <<<dimGrid, dimBlock, 0, stream->get_cuda_stream()>>>
                    (temp, this->from_row_indices.cast<float>(),
                        get_rows(), get_columns());

                this->from_column_indices.cast<float>().copy_to(temp, stream);
                cudaSetDevice(device_id);
                transpose_matrix_parallel<float>
                    <<<dimGrid, dimBlock, 0, stream->get_cuda_stream()>>>
                    (temp, this->from_column_indices.cast<float>(),
                        get_rows(), get_columns());
            }

            for (auto pair : variables) {
                auto p = Pointer<float>(pair.second);
                p.copy_to(temp, stream);
                cudaSetDevice(device_id);
                transpose_matrix_parallel<float>
                    <<<dimGrid, dimBlock, 0, stream->get_cuda_stream()>>>
                    (temp, p, get_rows(), get_columns());
            }
            temp.free();
            device_synchronize();
            device_check_error("Failed to transpose weight matrix!");
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
    device_check_error("Failed to transfer attributes to device!");
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
    std::vector<BasePointer*> pointers = { &weights };
    if (sparse) {
        pointers.push_back(&nonzero_counts);
        pointers.push_back(&from_row_indices);
        pointers.push_back(&from_column_indices);
    }
    if (transpose_flag)
        pointers.push_back(&weights_transposed);
    if (connection->second_order_host)
        pointers.push_back(&second_order_weights);
    for (auto pair : variables) pointers.push_back(pair.second);
    return pointers;
}

std::map<PointerKey, BasePointer*> WeightMatrix::get_pointer_map() {
    std::map<PointerKey, BasePointer*> pointers;
    pointers[
        PointerKey(connection->id, "weights", weights.get_bytes())] = &weights;

    if (sparse) {
        pointers[PointerKey(connection->id, "nonzero counts",
            nonzero_counts.get_bytes())] = &nonzero_counts;
        pointers[PointerKey(connection->id, "from row indices",
            from_row_indices.get_bytes())] = &from_row_indices;
        pointers[PointerKey(connection->id, "from column indices",
            from_column_indices.get_bytes())] = &from_column_indices;
    }
    if (transpose_flag)
        pointers[PointerKey(connection->id, "weights transposed",
            weights_transposed.get_bytes())] = &weights_transposed;
    if (connection->second_order_host)
        pointers[PointerKey(connection->id, "second order weights",
            second_order_weights.get_bytes())] = &second_order_weights;

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
    auto full_from_row_indices = Pointer<int>(num_weights, -1);
    auto full_from_column_indices = Pointer<int>(num_weights, -1);
    auto full_to_row_indices = Pointer<int>(num_weights, -1);
    auto full_to_column_indices = Pointer<int>(num_weights, -1);
    auto used = Pointer<int>(num_weights, 0);
    get_indices(this, used,
        full_from_row_indices, full_from_column_indices,
        full_to_row_indices, full_to_column_indices);
    full_to_row_indices.free();
    full_to_column_indices.free();

    // Compute nonzero weight counts and sparse matrix size
    int max_nonzero = 0;
    for (int row = 0 ; row < rows ; ++row) {
        int nonzero = 0;
        for (int col = 0 ; col < columns ; ++col) {
            int index = row*columns + col;
            nonzero += (weights[index] != 0.0 and used[index]);
        }
        max_nonzero = MAX(max_nonzero, nonzero);
    }
    int sparse_num_weights = max_nonzero * connection->to_layer->size;

    // Ensure nonzero size
    if (sparse_num_weights == 0)
        LOG_ERROR(
            "Error in weight config for " + connection->str() + ":\n" +
            "    Attempted to sparsify empty matrix!");
    else if (sparse_num_weights == num_weights) return;

    // Create index matrices (padded with -1) and nonzero counts
    this->from_row_indices = Pointer<int>(sparse_num_weights, -1);
    this->from_column_indices = Pointer<int>(sparse_num_weights, -1);
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
                from_row_indices[new_index]
                    = full_from_row_indices[old_index];
                from_column_indices[new_index]
                    = full_from_column_indices[old_index];
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

    // Free intermediate data
    used.free();
    full_from_row_indices.free();
    full_from_column_indices.free();

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
    Pointer<float> distances = Pointer<float>(num_weights, -1.0);
    get_distances(matrix, distances,
        from_spacing, to_spacing,
        x_offset, y_offset);

    callback(id, num_weights, target_matrix, distances.get());
    distances.free();
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
    int** p_to_p = (int**)synapse_data.pointer_to_pointer.get(); \
    int* from_row_is = (int*)p_to_p[0]; \
    int* from_col_is = (int*)p_to_p[1]; \
    int* to_row_indices = (int*)p_to_p[2]; \
    int* to_col_indices = (int*)p_to_p[3]; \
    int* used = (int*)p_to_p[4];

#define SET_INDICES(PREFIX) \
    from_row_is[weight_index] = PREFIX##from_row; \
    from_col_is[weight_index] = PREFIX##from_column; \
    to_row_indices[weight_index] = to_row; \
    to_col_indices[weight_index] = to_column; \
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

void get_indices(WeightMatrix *matrix, Pointer<int> used,
        Pointer<int> from_row_indices, Pointer<int> from_col_indices,
        Pointer<int> to_row_indices, Pointer<int> to_col_indices) {
    auto conn = matrix->connection;

    LOG_DEBUG("Retrieving to/from indices for : " + conn->str());

    // Retrieve appropriate kernel
    Kernel<SYNAPSE_ARGS> indices_kernel;
    try {
        switch(conn->get_type()) {
            case CONVERGENT:
                indices_kernel = get_indices_kernel_wrap_convergent();
                break;
            case DIVERGENT:
                indices_kernel = get_indices_kernel_wrap_divergent();
                break;
            default:
                indices_kernel = indices_kernel_map.at(conn->get_type());
                break;
        }
    } catch (std::out_of_range) {
        LOG_ERROR("Unrecognized connection type!");
    }

    auto res_man = ResourceManager::get_instance();
    Pointer<void*> p_to_p = Pointer<void*>(5);

    // Do on host if the connection is sparse (requires matrix data)
    if (res_man->get_gpu_ids().size() == 0 or conn->get_type() == SPARSE) {
        p_to_p[0] = from_row_indices.get();
        p_to_p[1] = from_col_indices.get();
        p_to_p[2] = to_row_indices.get();
        p_to_p[3] = to_col_indices.get();
        p_to_p[4] = used.get();

        LOG_DEBUG("  Running serial kernel...");
        indices_kernel.run_serial(SynapseData(matrix, conn, p_to_p));
    } else {
#ifdef __CUDACC__
        int num_weights = conn->get_num_weights();
        DeviceID device_id = *res_man->get_devices().begin();
        DeviceID host_id = res_man->get_host_id();
        Stream *stream = res_man->get_default_stream(device_id);

        LOG_DEBUG("  Allocating device memory...");

        std::vector<Pointer<int>> host_pointers =
            { from_row_indices, from_col_indices,
              to_row_indices, to_col_indices, used };
        int num_pointers = host_pointers.size();

        // Set up pointers
        Pointer<int> device_pointers[num_pointers];
        for (int i = 0 ; i < num_pointers ; ++i) {
            device_pointers[i] =
                Pointer<int>::device_pointer(device_id, num_weights);
            p_to_p[i] = device_pointers[i].get_unsafe();
        }
        // Set used to 0
        device_pointers[num_pointers-1].set(0, false);
        p_to_p.transfer(device_id);

        LOG_DEBUG("  Running parallel kernel...");

        // Run parallel kernel
        dim3 threads = calc_threads(conn->to_layer->size);
        dim3 blocks = calc_blocks(conn->to_layer->size);
        indices_kernel.run_parallel(
            stream, blocks, threads, SynapseData(nullptr, conn, p_to_p));

        LOG_DEBUG("  Performing transposition...");

        // Transpose matrices
        Pointer<int> temp =
            Pointer<int>::device_pointer(device_id, num_weights);

        int transpose_cols = (conn->convolutional) ? 1 : conn->to_layer->size;
        int transpose_rows = num_weights / transpose_cols;
        blocks = calc_transpose_blocks(transpose_rows, transpose_cols);
        threads = calc_transpose_threads(transpose_rows, transpose_cols);

        // Perform transpositions
        for (int i = 0 ; i < num_pointers ; ++i) {
            cudaSetDevice(device_id);
            transpose_matrix_parallel<int>
                <<<blocks, threads, 0, stream->get_cuda_stream()>>>
                (device_pointers[i], temp, transpose_rows, transpose_cols);

            temp.copy_to(host_pointers[i], stream);
            device_synchronize();
            device_check_error("Failed to transpose weight matrix!");
            device_pointers[i].free();
        }
        temp.free();
#endif
    }

    p_to_p.free();
}

#define DISTANCES_EXTRACTIONS \
    float** p_to_p = (float**)synapse_data.pointer_to_pointer.get(); \
    float* distances = (float*)p_to_p[0]; \
    float* aux_values = (float*)p_to_p[1]; \
    float from_spacing = aux_values[0]; \
    float to_spacing = aux_values[1]; \
    float x_offset = aux_values[2]; \
    float y_offset = aux_values[3];

#define SET_DISTANCES(PREFIX) \
    float f_y = PREFIX##from_row * from_spacing; \
    float f_x = PREFIX##from_column * from_spacing; \
 \
    float t_y = to_row * to_spacing + y_offset; \
    float t_x = to_column * to_spacing + x_offset; \
 \
    distances[weight_index] = pow( \
        pow(t_x - f_x, 2) + pow(t_y - f_y, 2), \
        0.5);

CALC_ALL(distances_kernel,
    DISTANCES_EXTRACTIONS,
    ,
    SET_DISTANCES(),
)

CALC_CONVERGENT(distances_kernel_wrap_convergent,
    DISTANCES_EXTRACTIONS,
    ,
    SET_DISTANCES(pre_wrap_),
)
CALC_DIVERGENT(distances_kernel_wrap_divergent,
    DISTANCES_EXTRACTIONS,
    ,
    SET_DISTANCES(pre_wrap_),
)

void get_distances(WeightMatrix *matrix, Pointer<float> distances,
        float from_spacing, float to_spacing,
        float x_offset, float y_offset) {
    auto conn = matrix->connection;

    LOG_DEBUG("Retrieving distances for : " + conn->str());

    // Retrieve appropriate kernel
    Kernel<SYNAPSE_ARGS> distances_kernel;
    try {
        switch(conn->get_type()) {
            case CONVERGENT:
                distances_kernel = get_distances_kernel_wrap_convergent();
                break;
            case DIVERGENT:
                distances_kernel = get_distances_kernel_wrap_divergent();
                break;
            default:
                distances_kernel = distances_kernel_map.at(conn->get_type());
                break;
        }
    } catch (std::out_of_range) {
        LOG_ERROR("Unrecognized connection type!");
    }

    auto res_man = ResourceManager::get_instance();
    Pointer<void*> p_to_p = Pointer<void*>(5);
    Pointer<float> auxiliary = Pointer<float>(4);
    auxiliary[0] = from_spacing;
    auxiliary[1] = to_spacing;
    auxiliary[2] = x_offset;
    auxiliary[3] = y_offset;

    // Do on host if the connection is sparse (requires matrix data)
    if (res_man->get_gpu_ids().size() == 0 or conn->get_type() == SPARSE) {
        p_to_p[0] = distances.get();
        p_to_p[1] = auxiliary.get();

        LOG_DEBUG("  Running serial kernel...");
        distances_kernel.run_serial(SynapseData(matrix, conn, p_to_p));
    } else {
#ifdef __CUDACC__
        int num_weights = conn->get_num_weights();
        DeviceID device_id = *res_man->get_devices().begin();
        DeviceID host_id = res_man->get_host_id();
        Stream *stream = res_man->get_default_stream(device_id);

        LOG_DEBUG("  Allocating device memory...");

        // Set up pointers
        auto device_distances
            = Pointer<float>::device_pointer(device_id, num_weights);
        p_to_p[0] = device_distances.get_unsafe();

        auxiliary.transfer(device_id);
        p_to_p[1] = auxiliary.get_unsafe();

        p_to_p.transfer(device_id);

        LOG_DEBUG("  Running parallel kernel...");

        // Run parallel kernel
        dim3 threads = calc_threads(conn->to_layer->size);
        dim3 blocks = calc_blocks(conn->to_layer->size);
        distances_kernel.run_parallel(
            stream, blocks, threads, SynapseData(nullptr, conn, p_to_p));

        LOG_DEBUG("  Performing transposition...");

        // Transpose matrices
        Pointer<float> temp =
            Pointer<float>::device_pointer(device_id, num_weights);

        int transpose_cols = (conn->convolutional) ? 1 : conn->to_layer->size;
        int transpose_rows = num_weights / transpose_cols;
        blocks = calc_transpose_blocks(transpose_rows, transpose_cols);
        threads = calc_transpose_threads(transpose_rows, transpose_cols);

        // Perform transposition
        cudaSetDevice(device_id);
        transpose_matrix_parallel<float>
            <<<blocks, threads, 0, stream->get_cuda_stream()>>>
            (device_distances, temp, transpose_rows, transpose_cols);

        temp.copy_to(distances, stream);
        device_synchronize();
        device_check_error("Failed to transpose weight matrix!");
        device_distances.free();
        temp.free();
#endif
    }

    auxiliary.free();
    p_to_p.free();
}

#define DELAYS_EXTRACTIONS \
    void** p_to_p = (void**)synapse_data.pointer_to_pointer.get(); \
    int* delays = (int*)p_to_p[0]; \
    float* aux_values = (float*)p_to_p[1]; \
    float from_spacing = aux_values[0]; \
    float to_spacing = aux_values[1]; \
    float x_offset = aux_values[2]; \
    float y_offset = aux_values[3]; \
    float velocity = aux_values[4]; \
    int base_delay = (int)aux_values[5]; \
    bool cap_delay = aux_values[6] > 0.5;

#define SET_DELAYS(PREFIX) \
    float f_y = PREFIX##from_row * from_spacing; \
    float f_x = PREFIX##from_column * from_spacing; \
 \
    float t_y = to_row * to_spacing + y_offset; \
    float t_x = to_column * to_spacing + x_offset; \
 \
    float distance = pow( \
        pow(t_x - f_x, 2) + pow(t_y - f_y, 2), \
        0.5); \
    int delay = base_delay + (distance / velocity); \
    if (delay > 31 and not cap_delay) { \
        printf("BIT delays cannot be greater than 31!"); \
        assert(false); \
    } \
    delays[weight_index] = MIN(31, delay);

CALC_ALL(delays_kernel,
    DELAYS_EXTRACTIONS,
    ,
    SET_DELAYS(),
)

CALC_CONVERGENT(delays_kernel_wrap_convergent,
    DELAYS_EXTRACTIONS,
    ,
    SET_DELAYS(pre_wrap_),
)
CALC_DIVERGENT(delays_kernel_wrap_divergent,
    DELAYS_EXTRACTIONS,
    ,
    SET_DELAYS(pre_wrap_),
)

void get_delays(WeightMatrix *matrix, OutputType output_type,
        Pointer<int> delays,
        float from_spacing, float to_spacing,
        float x_offset, float y_offset,
        float velocity, bool cap_delay) {
    auto conn = matrix->connection;

    LOG_DEBUG("Initializing delays for : " + conn->str());
    if (output_type != BIT)
        LOG_ERROR("Only BIT output connections can have variable delays!");

    int base_delay = conn->delay;
    float num_weights = conn->get_num_weights();

    // Retrieve appropriate kernel
    Kernel<SYNAPSE_ARGS> delays_kernel;
    try {
        switch(conn->get_type()) {
            case CONVERGENT:
                delays_kernel = get_delays_kernel_wrap_convergent();
                break;
            case DIVERGENT:
                delays_kernel = get_delays_kernel_wrap_divergent();
                break;
            default:
                delays_kernel = delays_kernel_map.at(conn->get_type());
                break;
        }
    } catch (std::out_of_range) {
        LOG_ERROR("Unrecognized connection type!");
    }

    auto res_man = ResourceManager::get_instance();
    Pointer<void*> p_to_p = Pointer<void*>(5);
    Pointer<float> auxiliary = Pointer<float>(7);
    auxiliary[0] = from_spacing;
    auxiliary[1] = to_spacing;
    auxiliary[2] = x_offset;
    auxiliary[3] = y_offset;
    auxiliary[4] = velocity;
    auxiliary[5] = base_delay;
    auxiliary[6] = (cap_delay) ? 1.0 : 0.0;

    // Do on host if the connection is sparse (requires matrix data)
    if (res_man->get_gpu_ids().size() == 0 or conn->get_type() == SPARSE) {
        p_to_p[0] = (void*)delays.get();
        p_to_p[1] = (void*)auxiliary.get();

        LOG_DEBUG("  Running serial kernel...");
        delays_kernel.run_serial(SynapseData(matrix, conn, p_to_p));
    } else {
#ifdef __CUDACC__
        int num_weights = conn->get_num_weights();
        DeviceID device_id = *res_man->get_devices().begin();
        DeviceID host_id = res_man->get_host_id();
        Stream *stream = res_man->get_default_stream(device_id);

        LOG_DEBUG("  Allocating device memory...");

        // Set up pointers
        auto device_delays
            = Pointer<int>::device_pointer(device_id, num_weights);
        p_to_p[0] = device_delays.get_unsafe();

        auxiliary.transfer(device_id);
        p_to_p[1] = auxiliary.get_unsafe();

        p_to_p.transfer(device_id);

        LOG_DEBUG("  Running parallel kernel...");

        // Run parallel kernel
        dim3 threads = calc_threads(conn->to_layer->size);
        dim3 blocks = calc_blocks(conn->to_layer->size);
        delays_kernel.run_parallel(
            stream, blocks, threads, SynapseData(nullptr, conn, p_to_p));

        LOG_DEBUG("  Performing transposition...");

        // Transpose matrices
        Pointer<int> temp =
            Pointer<int>::device_pointer(device_id, num_weights);

        int transpose_cols = (conn->convolutional) ? 1 : conn->to_layer->size;
        int transpose_rows = num_weights / transpose_cols;
        blocks = calc_transpose_blocks(transpose_rows, transpose_cols);
        threads = calc_transpose_threads(transpose_rows, transpose_cols);

        // Perform transposition
        cudaSetDevice(device_id);
        transpose_matrix_parallel<int>
            <<<blocks, threads, 0, stream->get_cuda_stream()>>>
            (device_delays, temp, transpose_rows, transpose_cols);

        temp.copy_to(delays, stream);
        device_synchronize();
        device_check_error("Failed to transpose weight matrix!");
        device_delays.free();
        temp.free();
#endif
    }

    auxiliary.free();
    p_to_p.free();
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
