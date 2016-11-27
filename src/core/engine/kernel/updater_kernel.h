#ifndef updater_kernel_h
#define updater_kernel_h

#include "engine/kernel/kernel.h"
#include "util/parallel.h"
#include "util/constants.h"

class ConnectionData;

/* Updaters are responsible for updating connection weights */
typedef void(*UPDATER)(ConnectionData);
void get_updater(UPDATER *dest, ConnectionType conn_type);

GLOBAL void update_fully_connected(ConnectionData conn_data);
GLOBAL void update_one_to_one(ConnectionData conn_data);
GLOBAL void update_convergent(ConnectionData conn_data);

#endif
