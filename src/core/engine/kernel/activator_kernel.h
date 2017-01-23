#ifndef activator_kernel_h
#define activator_kernel_h

#include "engine/kernel/kernel.h"

/* Activators are responsible for performing connection computation */
KERNEL get_activator_kernel(ConnectionType conn_type);

GLOBAL void activate_fully_connected(KernelData kernel_data);
GLOBAL void activate_one_to_one(KernelData kernel_data);
GLOBAL void activate_convergent(KernelData kernel_data);
GLOBAL void calc_internal(int size, float *src, float *dst);

#endif
