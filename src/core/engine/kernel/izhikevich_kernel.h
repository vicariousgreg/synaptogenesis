#ifndef izhikevich_kernel_h
#define izhikevich_kernel_h

#include "state/izhikevich_attributes.h"
#include "util/parallel.h"

/* Izhikevich voltage update function */
GLOBAL void izhikevich(IzhikevichAttributes *att, int start_index, int count);

/* Spike update function */
GLOBAL void calc_spikes(IzhikevichAttributes *att,
    int start_index, int count, int num_neurons);

#endif
