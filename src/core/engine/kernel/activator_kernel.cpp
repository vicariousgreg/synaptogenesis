#include "engine/kernel/activator_kernel.h"
#include "state/attributes.h"
#include "util/error_manager.h"

/******************************************************************************/
/********************** CONNECTION ACTIVATOR KERNELS **************************/
/******************************************************************************/

KERNEL get_base_activator_kernel(ConnectionType conn_type) {
    switch (conn_type) {
        case FULLY_CONNECTED:
            return activate_fully_connected;
        case ONE_TO_ONE:
            return activate_one_to_one;
        case CONVERGENT:
        case CONVOLUTIONAL:
            return activate_convergent;
        default:
            ErrorManager::get_instance()->log_error(
                "Unimplemented connection type!");
    }
}

/* Vanilla versions of activator functions */
ACTIVATE_FULLY_CONNECTED(activate_fully_connected , , );
ACTIVATE_ONE_TO_ONE(activate_one_to_one , , );
ACTIVATE_CONVERGENT(activate_convergent , , );

/* Dendritic tree internal computation */
#ifdef PARALLEL
GLOBAL void calc_internal(int size, float *src, float *dst, bool clear) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        dst[index] += src[index];
        if (clear) src[index] = 0.0;
    }
}
#else
GLOBAL void calc_internal(int size, float *src, float *dst, bool clear) {
    if (clear) {
        for (int index = 0 ; index < size ; ++index) {
            dst[index] += src[index];
            src[index] = 0.0;
        }
    } else
        for (int index = 0 ; index < size ; ++index)
            dst[index] += src[index];
}
#endif
