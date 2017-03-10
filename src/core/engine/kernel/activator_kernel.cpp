#include "engine/kernel/activator_kernel.h"
#include "state/attributes.h"
#include "util/error_manager.h"

/******************************************************************************/
/********************** CONNECTION ACTIVATOR KERNELS **************************/
/******************************************************************************/

/* Vanilla versions of activator functions */
ACTIVATE_FULLY_CONNECTED(activate_fully_connected , , );
ACTIVATE_ONE_TO_ONE(activate_one_to_one , , );
ACTIVATE_CONVERGENT(activate_convergent , , );

Kernel<SYNAPSE_ARGS>get_base_activator_kernel(ConnectionType conn_type) {
    switch (conn_type) {
        case FULLY_CONNECTED:
            return get_activate_fully_connected();
        case ONE_TO_ONE:
            return get_activate_one_to_one();
        case CONVERGENT:
        case CONVOLUTIONAL:
            return get_activate_convergent();
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}

/* Dendritic tree internal computation */
inline void calc_internal_SERIAL(int size, Pointer<float> src_ptr,
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
inline GLOBAL void calc_internal_PARALLEL(int size, Pointer<float> src_ptr,
        Pointer<float> dst_ptr, bool clear=false) {
    float* src = src_ptr.get();
    float* dst = dst_ptr.get();
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        dst[index] += src[index];
        if (clear) src[index] = 0.0;
    }
}
Kernel<int, Pointer<float>, Pointer<float>, bool> get_calc_internal() {
    return Kernel<int, Pointer<float>, Pointer<float>, bool>(
        calc_internal_SERIAL, calc_internal_PARALLEL);
}
#else
Kernel<int, Pointer<float>, Pointer<float>, bool> get_calc_internal() {
    return Kernel<int, Pointer<float>, Pointer<float>, bool>(
        calc_internal_SERIAL);
}
#endif
