#ifndef izhikevich_kernel_h
#define izhikevich_kernel_h

#include "state/izhikevich_attributes.h"
#include "util/parallel.h"

/* Izhikevich voltage update and spike calculation */
GLOBAL void iz_update_attributes(IzhikevichAttributes *att,
    int start_index, int count, int num_neurons);

#endif
