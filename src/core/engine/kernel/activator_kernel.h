#ifndef activator_kernel_h
#define activator_kernel_h

#include "engine/kernel/kernel.h"

/* Activators are responsible for performing connection computation */

// Vanilla activators
KERNEL get_activator_kernel(ConnectionType conn_type);
GLOBAL void activate_fully_connected(KernelData kernel_data);
GLOBAL void activate_one_to_one(KernelData kernel_data);
GLOBAL void activate_convergent(KernelData kernel_data);

// Trace activators
KERNEL get_activator_kernel_trace(ConnectionType conn_type);
GLOBAL void activate_fully_connected_trace(KernelData kernel_data);
GLOBAL void activate_one_to_one_trace(KernelData kernel_data);
GLOBAL void activate_convergent_trace(KernelData kernel_data);

// Internal activator
GLOBAL void calc_internal(int size, float *src, float *dst, bool clear=false);

#endif
