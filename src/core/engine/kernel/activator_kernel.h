#ifndef activator_kernel_h
#define activator_kernel_h

#include "engine/kernel/kernel.h"
#include "engine/kernel/connection_kernel.h"
#include "util/parallel.h"

class KernelData;

/* Activators are responsible for performing connection computation */
typedef void(*ACTIVATOR)(KernelData);
void get_activator(ACTIVATOR *dest, ConnectionType conn_type);

GLOBAL void calc_fully_connected(KernelData kernel_data);
GLOBAL void calc_one_to_one(KernelData kernel_data);
GLOBAL void calc_convergent(KernelData kernel_data);

#endif
