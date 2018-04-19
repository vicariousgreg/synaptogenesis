#include "engine/kernel/kernel.h"
#include "engine/kernel/synapse_kernel.h"
#include "state/weight_matrix.h"
#include "util/tools.h"
#include "util/parallel.h"
#include "util/transpose.h"

/******************************************************************************/
/************************* ACTIVATOR TOOL KERNELS *****************************/
/******************************************************************************/

/* Clears input data */
void set_data_SERIAL(float val, Pointer<float> ptr, int count, bool overwrite) {
    float* data = ptr.get();

    if (overwrite)
        for (int nid = 0; nid < count; ++nid)
            data[nid] = val;
    else
        for (int nid = 0; nid < count; ++nid)
            data[nid] += val;
}
#ifdef __CUDACC__
GLOBAL void set_data_PARALLEL(float val, Pointer<float> ptr,
        int count, bool overwrite) {
    float* data = ptr.get();

    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count)
        if (overwrite) data[nid] = val;
        else           data[nid] += val;
}
#else
GLOBAL void set_data_PARALLEL(float val, Pointer<float> ptr, int count) { }
#endif
Kernel<float, Pointer<float>, int, bool> get_set_data() {
    return Kernel<float, Pointer<float>, int, bool>(
        set_data_SERIAL, set_data_PARALLEL);
}

/* Randomizes input data using Uniform Distribution */
void randomize_data_uniform_SERIAL(Pointer<float> ptr,
        int count, float min, float max, bool overwrite) {
    std::uniform_real_distribution<float> distribution(min, max);
    float* data = ptr.get();

    if (overwrite)
        for (int nid = 0; nid < count; ++nid)
            data[nid] = distribution(generator);
    else
        for (int nid = 0; nid < count; ++nid)
            data[nid] += distribution(generator);
}
#ifdef __CUDACC__
GLOBAL void randomize_data_uniform_PARALLEL(Pointer<float> ptr,
        int count, float min, float max, bool overwrite) {
    float* data = ptr.get();

    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count) {
        float val = (curand_uniform(&cuda_rand_states[nid]) * (max-min)) + min;
        if (overwrite) data[nid] = val;
        else           data[nid] += val;
    }
}
#else
GLOBAL void randomize_data_uniform_PARALLEL(Pointer<float> ptr,
        int count, float min, float max, bool overwrite) { }
#endif
Kernel<Pointer<float>, int, float, float, bool>
        get_randomize_data_uniform() {
    return Kernel<Pointer<float>, int, float, float, bool>(
        randomize_data_uniform_SERIAL, randomize_data_uniform_PARALLEL);
}

/* Randomizes input data using Normal Distribution */
void randomize_data_normal_SERIAL(Pointer<float> ptr,
        int count, float mean, float std_dev, bool overwrite) {
    std::normal_distribution<float> distribution(mean, std_dev);
    float* data = ptr.get();

    if (overwrite)
        for (int nid = 0; nid < count; ++nid)
            data[nid] = distribution(generator);
    else
        for (int nid = 0; nid < count; ++nid)
            data[nid] += distribution(generator);
}
#ifdef __CUDACC__
GLOBAL void randomize_data_normal_PARALLEL(Pointer<float> ptr,
        int count, float mean, float std_dev, bool overwrite) {
    float* data = ptr.get();

    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count) {
        float val = (curand_normal(&cuda_rand_states[nid]) * std_dev) + mean;
        if (overwrite) data[nid] = val;
        else           data[nid] += val;
    }
}
#else
GLOBAL void randomize_data_normal_PARALLEL(Pointer<float> ptr,
        int count, float mean, float std_dev, bool overwrite) { }
#endif
Kernel<Pointer<float>, int, float, float, bool>
        get_randomize_data_normal() {
    return Kernel<Pointer<float>, int, float, float, bool>(
        randomize_data_normal_SERIAL, randomize_data_normal_PARALLEL);
}

/* Randomizes input data using Poisson Point Process */
void randomize_data_poisson_SERIAL(Pointer<float> ptr, int count, float val,
        float rate, bool overwrite, Pointer<float> random_rates) {
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    float* data = ptr.get();
    float* rrates = random_rates.get();
    bool random = rrates != nullptr;

    if (overwrite)
        for (int nid = 0; nid < count; ++nid)
            data[nid] =
                (distribution(generator) < ((random) ? rrates[nid] : rate))
                ? val : 0.0;
    else
        for (int nid = 0; nid < count; ++nid)
            if (distribution(generator) < ((random) ? rrates[nid] : rate))
                data[nid] += val;
}
#ifdef __CUDACC__
GLOBAL void randomize_data_poisson_PARALLEL(Pointer<float> ptr, int count,
        float val, float rate, bool overwrite, Pointer<float> random_rates) {
    float* data = ptr.get();
    float* rrates = random_rates.get();
    bool random = rrates != nullptr;

    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count) {
        if (overwrite)
            data[nid] =
                (curand_uniform(&cuda_rand_states[nid])
                        < ((random) ? rrates[nid] : rate))
                    ? val
                    : 0.0;
        else if (curand_uniform(&cuda_rand_states[nid])
                        < ((random) ? rrates[nid] : rate))
            data[nid] += val;
    }
}
#else
GLOBAL void randomize_data_poisson_PARALLEL(Pointer<float> ptr, int count,
        float val, float rate, bool overwrite, Pointer<float> random_rates) { }
#endif
Kernel<Pointer<float>, int, float, float, bool, Pointer<float>>
        get_randomize_data_poisson() {
    return Kernel<Pointer<float>, int, float, float, bool, Pointer<float>>(
        randomize_data_poisson_SERIAL, randomize_data_poisson_PARALLEL);
}

/* Dendritic tree internal computation */
void calc_internal_SERIAL(int size, Pointer<float> src_ptr,
        Pointer<float> dst_ptr, AGGREGATOR aggregate, float trail_value) {
    float* src = src_ptr.get();
    float* dst = dst_ptr.get();
    for (int index = 0 ; index < size ; ++index) {
        dst[index] = aggregate(dst[index], src[index]);
        src[index] = trail_value;
    }
}
#ifdef __CUDACC__
GLOBAL void calc_internal_PARALLEL(int size, Pointer<float> src_ptr,
        Pointer<float> dst_ptr, AGGREGATOR aggregate, float trail_value) {
    float* src = src_ptr.get();
    float* dst = dst_ptr.get();
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        dst[index] = aggregate(dst[index], src[index]);
        src[index] = trail_value;
    }
}
#else
GLOBAL void calc_internal_PARALLEL(int size, Pointer<float> src_ptr,
        Pointer<float> dst_ptr, float trail_value) { }
#endif
Kernel<int, Pointer<float>, Pointer<float>, AGGREGATOR,
        float> get_calc_internal() {
    return Kernel<int, Pointer<float>, Pointer<float>, AGGREGATOR, float>(
        calc_internal_SERIAL, calc_internal_PARALLEL);
}

void transpose_matrix_serial(
        const Pointer<float> idata, Pointer<float> odata,
        const int original_rows, const int original_columns) {
    float* in = idata.get();
    float* out = odata.get();

    for (int i = 0 ; i < original_rows ; ++i)
        for (int j = 0 ; j < original_columns ; ++j)
            out[(j*original_rows) + i] = in[(i*original_columns) + j];
}

Kernel<const Pointer<float>, Pointer<float>,
        const int, const int> get_transposer() {
#ifdef __CUDACC__
    return Kernel<const Pointer<float>, Pointer<float>, const int, const int>(
        transpose_matrix_serial,transpose_matrix_parallel<float>);
#else
    return Kernel<const Pointer<float>, Pointer<float>, const int, const int>(
        transpose_matrix_serial);
#endif
}

/******************************************************************************/
/********************** CONNECTION ACTIVATOR KERNELS **************************/
/******************************************************************************/

/* Vanilla versions of activator functions */
ACTIVATE_ALL(activate_base , , );

/* Second order */
ACTIVATE_ALL_SECOND_ORDER(activate_base_second_order , , );

Kernel<SYNAPSE_ARGS> get_base_activator_kernel(Connection *conn) {
    // Handle second order convolutional connections
    if (conn->convolutional and conn->second_order_slave) {
        if (conn->get_type() == CONVERGENT)
            return get_activate_base_second_order_convergent_convolutional();
        else if (conn->get_type() == DIVERGENT)
            return get_activate_base_second_order_divergent_convolutional();
    }

    // Handle all other connections
    // Use second order kernels for slave connections
    try {
        if (conn->second_order_slave)
            return activate_base_second_order_map.at(conn->get_type());
        else
            return activate_base_map.at(conn->get_type());
    } catch(std::out_of_range) { }

    LOG_ERROR(
        "Attempted to retrieve base activator kernel for "
        "unimplemented connection type!");
}
