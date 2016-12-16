#ifndef izhikevich_kernel_h
#define izhikevich_kernel_h

#include "state/izhikevich_attributes.h"
#include "util/parallel.h"

/* Izhikevich voltage update function */
GLOBAL void iz_update_state(IzhikevichAttributes *att,
    int start_index, int count);

/* Spike update function */
GLOBAL void iz_update_output(IzhikevichAttributes *att,
    int start_index, int count, int num_neurons);

#endif
