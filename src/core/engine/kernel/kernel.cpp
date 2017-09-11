#include "engine/kernel/kernel.h"
#include "engine/kernel/synapse_kernel.h"

/******************************************************************************/
/************************* ACTIVATOR TOOL KERNELS *****************************/
/******************************************************************************/

/* Clears input data */
void set_data_SERIAL(float val, Pointer<float> ptr, int count) {
    float* data = ptr.get();

    for (int nid = 0; nid < count; ++nid)
        data[nid] = val;
}
#ifdef __CUDACC__
GLOBAL void set_data_PARALLEL(float val, Pointer<float> ptr, int count) {
    float* data = ptr.get();

    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count)
        data[nid] = val;
}
#else
GLOBAL void set_data_PARALLEL(float val, Pointer<float> ptr, int count) { }
#endif
Kernel<float, Pointer<float>, int> get_set_data() {
    return Kernel<float, Pointer<float>, int>(
        set_data_SERIAL, set_data_PARALLEL);
}

/* Randomizes input data using Normal Distribution */
void randomize_data_normal_SERIAL(Pointer<float> ptr,
        int count, float mean, float std_dev, bool init) {
    std::default_random_engine generator(time(0));
    std::normal_distribution<float> distribution(mean, std_dev);
    float* data = ptr.get();

    if (init)
        for (int nid = 0; nid < count; ++nid)
            data[nid] = distribution(generator);
    else
        for (int nid = 0; nid < count; ++nid)
            data[nid] += distribution(generator);
}
#ifdef __CUDACC__
GLOBAL void randomize_data_normal_PARALLEL(Pointer<float> ptr,
        int count, float mean, float std_dev, bool init) {
    float* data = ptr.get();

    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count) {
        float val = (curand_normal(&cuda_rand_states[nid]) * std_dev) + mean;
        if (init) data[nid] = val;
        else      data[nid] += val;
    }
}
#else
GLOBAL void randomize_data_normal_PARALLEL(Pointer<float> ptr,
        int count, float mean, float std_dev, bool init) { }
#endif
Kernel<Pointer<float>, int, float, float, bool> get_randomize_data_normal() {
    return Kernel<Pointer<float>, int, float, float, bool>(
        randomize_data_normal_SERIAL, randomize_data_normal_PARALLEL);
}

/* Randomizes input data using Poisson Point Process */
void randomize_data_poisson_SERIAL(Pointer<float> ptr,
        int count, float val, float rate, bool init) {
    std::default_random_engine generator(time(0));
    std::uniform_real_distribution<float> distribution(0.0, 1.0);
    float* data = ptr.get();

    if (init)
        for (int nid = 0; nid < count; ++nid)
            data[nid] = (distribution(generator) < rate) ? val : 0.0;
    else
        for (int nid = 0; nid < count; ++nid)
            if (distribution(generator) < rate)
                data[nid] += val;
}
#ifdef __CUDACC__
GLOBAL void randomize_data_poisson_PARALLEL(Pointer<float> ptr,
        int count, float val, float rate, bool init) {
    float* data = ptr.get();


    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count) {
        if (init)
            data[nid] =
                (curand_uniform(&cuda_rand_states[nid]) < rate)
                ? val : 0.0;
        else if (curand_uniform(&cuda_rand_states[nid]) < rate)
            data[nid] += val;
    }
}
#else
GLOBAL void randomize_data_poisson_PARALLEL(Pointer<float> ptr,
        int count, float val, float rate, bool init) { }
#endif
Kernel<Pointer<float>, int, float, float, bool> get_randomize_data_poisson() {
    return Kernel<Pointer<float>, int, float, float, bool>(
        randomize_data_poisson_SERIAL, randomize_data_poisson_PARALLEL);
}

/* Dendritic tree internal computation */
void calc_internal_SERIAL(int size, Pointer<float> src_ptr,
        Pointer<float> dst_ptr, bool clear=false) {
    float* src = src_ptr.get();
    float* dst = dst_ptr.get();
    if (clear) {
        for (int index = 0 ; index < size ; ++index) {
            dst[index] += src[index];
            src[index] = 0.0;
        }
    } else
        for (int index = 0 ; index < size ; ++index)
            dst[index] += src[index];
}
#ifdef __CUDACC__
GLOBAL void calc_internal_PARALLEL(int size, Pointer<float> src_ptr,
        Pointer<float> dst_ptr, bool clear=false) {
    float* src = src_ptr.get();
    float* dst = dst_ptr.get();
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        dst[index] += src[index];
        if (clear) src[index] = 0.0;
    }
}
#else
GLOBAL void calc_internal_PARALLEL(int size, Pointer<float> src_ptr,
        Pointer<float> dst_ptr, bool clear=false) { }
#endif
Kernel<int, Pointer<float>, Pointer<float>, bool> get_calc_internal() {
    return Kernel<int, Pointer<float>, Pointer<float>, bool>(
        calc_internal_SERIAL, calc_internal_PARALLEL);
}

/******************************************************************************/
/********************** CONNECTION ACTIVATOR KERNELS **************************/
/******************************************************************************/

/* Vanilla versions of activator functions */
ACTIVATE_ALL(activate_base , , );

/* Second order */
ACTIVATE_ALL_SECOND_ORDER(activate_base_second_order , , );

Kernel<SYNAPSE_ARGS> get_base_activator_kernel(Connection *conn) {
    switch (conn->type) {
        case FULLY_CONNECTED:
            return (conn->second_order_slave)
                ? get_activate_base_second_order_fully_connected()
                : get_activate_base_fully_connected();
        case SUBSET:
            return (conn->second_order_slave)
                ? get_activate_base_second_order_subset()
                : get_activate_base_subset();
        case ONE_TO_ONE:
            return (conn->second_order_slave)
                ? get_activate_base_second_order_one_to_one()
                : get_activate_base_one_to_one();
        case CONVERGENT:
            return (conn->second_order_slave)
                ? get_activate_base_second_order_convergent()
                : get_activate_base_convergent();
        case CONVOLUTIONAL:
            return (conn->second_order_slave)
                ? get_activate_base_second_order_convolutional()
                : get_activate_base_convergent();
        case DIVERGENT:
            return (conn->second_order_slave)
                ? get_activate_base_second_order_divergent()
                : get_activate_base_divergent();
        default:
            ErrorManager::get_instance()->log_error(
                "Attempted to retrieve base activator kernel for "
                "unimplemented connection type!");
    }
}
