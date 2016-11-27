#ifndef activator_kernel_h
#define activator_kernel_h

#include "engine/kernel/kernel.h"
#include "util/parallel.h"
#include "util/constants.h"

class ConnectionData;

/* Activators are responsible for performing connection computation */
typedef void(*ACTIVATOR)(ConnectionData);
void get_activator(ACTIVATOR *dest, ConnectionType conn_type);

GLOBAL void calc_fully_connected(ConnectionData conn_data);
GLOBAL void calc_one_to_one(ConnectionData conn_data);
GLOBAL void calc_convergent(ConnectionData conn_data);

#endif
