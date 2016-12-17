#ifndef updater_kernel_h
#define updater_kernel_h

#include "engine/kernel/kernel.h"

/* Updaters are responsible for updating connection weights */
void get_updater_kernel(KERNEL *dest, ConnectionType conn_type);

GLOBAL void update_fully_connected(KernelData kernel_data);
GLOBAL void update_one_to_one(KernelData kernel_data);
GLOBAL void update_convergent(KernelData kernel_data);

#endif
