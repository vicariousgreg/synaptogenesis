#ifndef updater_kernel_h
#define updater_kernel_h

#include "engine/kernel/kernel.h"
#include "engine/kernel/connection_kernel.h"
#include "util/parallel.h"

class KernelData;

/* Updaters are responsible for updating connection weights */
typedef void(*UPDATER)(KernelData);
void get_updater(UPDATER *dest, ConnectionType conn_type);

GLOBAL void update_fully_connected(KernelData kernel_data);
GLOBAL void update_one_to_one(KernelData kernel_data);
GLOBAL void update_convergent(KernelData kernel_data);

#endif
