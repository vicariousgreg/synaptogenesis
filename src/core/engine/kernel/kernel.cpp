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

/* Randomizes input data */
void randomize_data_SERIAL(Pointer<float> ptr,
        int count, float mean, float std_dev, bool init) {
    std::default_random_engine generator(time(0));
    std::normal_distribution<double> distribution(mean, std_dev);
    float* data = ptr.get();

    if (init)
        for (int nid = 0; nid < count; ++nid)
            data[nid] = distribution(generator);
    else
        for (int nid = 0; nid < count; ++nid)
            data[nid] += distribution(generator);
}
#ifdef __CUDACC__
GLOBAL void randomize_data_PARALLEL(Pointer<float> ptr,
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
GLOBAL void randomize_data_PARALLEL(Pointer<float> ptr,
        int count, float mean, float std_dev, bool init) { }
#endif
Kernel<Pointer<float>, int, float, float, bool> get_randomize_data() {
    return Kernel<Pointer<float>, int, float, float, bool>(
        randomize_data_SERIAL, randomize_data_PARALLEL);
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

/* Dendritic tree second order internal computation */
void calc_internal_second_order_SERIAL(int from_size, int to_size,
        Pointer<float> src_ptr, Pointer<float> dst_ptr) {
    float* src = src_ptr.get();
    float* dst = dst_ptr.get();

    for (int to_index = 0 ; to_index < to_size ; ++to_index) {
        float sum = 0.0;
        for (int from_index = 0 ; from_index < from_size ; ++from_index)
            sum += src[to_index * from_size + from_index];
        dst[to_index] = sum;
    }
}
#ifdef __CUDACC__
GLOBAL void calc_internal_second_order_PARALLEL(int from_size, int to_size,
        Pointer<float> src_ptr, Pointer<float> dst_ptr) {
    float* src = src_ptr.get();
    float* dst = dst_ptr.get();
    int to_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (to_index < to_size) {
        float sum = 0.0;
        for (int from_index = 0 ; from_index < from_size ; ++from_index)
            sum += src[from_index * to_size + to_index];
        dst[to_index] = sum;
    }
}
#else
GLOBAL void calc_internal_second_order_PARALLEL(int from_size, int to_size,
        Pointer<float> src_ptr, Pointer<float> dst_ptr) { }
#endif
Kernel<int, int, Pointer<float>, Pointer<float> >
get_calc_internal_second_order() {
    return Kernel<int, int, Pointer<float>, Pointer<float> >(
        calc_internal_second_order_SERIAL, calc_internal_second_order_PARALLEL);
}

/******************************************************************************/
/********************** CONNECTION ACTIVATOR KERNELS **************************/
/******************************************************************************/

/* Vanilla versions of activator functions */
ACTIVATE_ALL(activate_base , , );

/* Second order */
ACTIVATE_ALL_SECOND_ORDER(activate_base_second_order , , );

Kernel<SYNAPSE_ARGS> get_base_activator_kernel(
        ConnectionType conn_type, bool second_order) {
    switch (conn_type) {
        case FULLY_CONNECTED:
            return (second_order)
                ? get_activate_base_second_order_fully_connected()
                : get_activate_base_fully_connected();
        case ONE_TO_ONE:
            return (second_order)
                ? get_activate_base_second_order_one_to_one()
                : get_activate_base_one_to_one();
        case CONVERGENT:
        case CONVOLUTIONAL:
            return (second_order)
                ? get_activate_base_second_order_convergent()
                : get_activate_base_convergent();
        case DIVERGENT:
            return (second_order)
                ? get_activate_base_second_order_divergent()
                : get_activate_base_divergent();
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}
