#ifndef updater_kernel_h
#define updater_kernel_h

#include "engine/kernel/kernel.h"

/* Updaters are responsible for updating connection weights */
KERNEL get_updater_kernel(ConnectionType conn_type);

GLOBAL void update_fully_connected(const KernelData kernel_data);
GLOBAL void update_one_to_one(const KernelData kernel_data);
GLOBAL void update_convergent(const KernelData kernel_data);
GLOBAL void update_convolutional(const KernelData kernel_data);

#endif
