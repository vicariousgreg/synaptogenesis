#include "engine/kernel/kernel.h"

/******************************************************************************/
/************************* ACTIVATOR TOOL KERNELS *****************************/
/******************************************************************************/

/* Clears input data */
void clear_data_SERIAL(Pointer<float> ptr, int count) {
    float* data = ptr.get();

    for (int nid = 0; nid < count; ++nid)
        data[nid] = 0.0;
}
#ifdef __CUDACC__
GLOBAL void clear_data_PARALLEL(Pointer<float> ptr, int count) {
    float* data = ptr.get();

    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count)
        data[nid] = 0.0;
}
#else
GLOBAL void clear_data_PARALLEL(Pointer<float> ptr, int count) { }
#endif
Kernel<Pointer<float>, int> get_clear_data() {
    return Kernel<Pointer<float>, int>(
        clear_data_SERIAL, clear_data_PARALLEL);
}

/* Randomizes input data */
void randomize_data_SERIAL(Pointer<float> ptr,
        int count, float max, bool init) {
    float* data = ptr.get();

    if (init)
        for (int nid = 0; nid < count; ++nid)
            data[nid] = fRand(0.0, max);
    else
        for (int nid = 0; nid < count; ++nid)
            data[nid] += fRand(0.0, max);
}
#ifdef __CUDACC__
GLOBAL void randomize_data_PARALLEL(Pointer<float> ptr,
        int count, float max, bool init) {
    float* data = ptr.get();

    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count) {
        float val = curand_uniform(&cuda_rand_states[nid]) * max;
        if (init)
            data[nid] = val;
        else
            data[nid] += val;
    }
}
#else
GLOBAL void randomize_data_PARALLEL(Pointer<float> ptr,
        int count, float max, bool init) { }
#endif
Kernel<Pointer<float>, int, float, bool> get_randomize_data() {
    return Kernel<Pointer<float>, int, float, bool>(
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

/******************************************************************************/
/********************** CONNECTION ACTIVATOR KERNELS **************************/
/******************************************************************************/

/* Vanilla versions of activator functions */
ACTIVATE_FULLY_CONNECTED(activate_fully_connected_base , , );
ACTIVATE_ONE_TO_ONE(activate_one_to_one_base , , );
ACTIVATE_CONVERGENT(activate_convergent_base , , );
ACTIVATE_DIVERGENT(activate_divergent_base , , );

Kernel<SYNAPSE_ARGS> get_base_activator_kernel(ConnectionType conn_type) {
    switch (conn_type) {
        case FULLY_CONNECTED:
            return get_activate_fully_connected_base();
        case ONE_TO_ONE:
            return get_activate_one_to_one_base();
        case CONVERGENT:
        case CONVOLUTIONAL:
            return get_activate_convergent_base();
        case DIVERGENT:
            return get_activate_divergent_base();
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}
