#ifndef updater_kernel_h
#define updater_kernel_h

#include "engine/kernel/kernel.h"

/* Updaters are responsible for updating connection weights */

#define UPDATE_FULLY_CONNECTED(FUNC_NAME, UPDATE_EXT, UPDATE_CALC) \
CALC_FULLY_CONNECTED(FUNC_NAME, \
    /* EXTRACTIONS */ \
    UPDATE_EXT;, \
 \
    /* NEURON_PRE
     * Retrieve input sum */ \
    float sum = inputs[to_index];, \
 \
    /* WEIGHT_OP */ \
    UPDATE_CALC;, \
 \
    /* NEURON_POST
     * no_op */ \
    ; \
)

#define UPDATE_ONE_TO_ONE(FUNC_NAME, UPDATE_EXT, UPDATE_CALC) \
CALC_ONE_TO_ONE(FUNC_NAME, \
    /* EXTRACTIONS */ \
    UPDATE_EXT;, \
 \
    /* WEIGHT_OP */ \
    UPDATE_CALC; \
)

#define UPDATE_CONVERGENT(FUNC_NAME, UPDATE_EXT, UPDATE_CALC) \
CALC_CONVERGENT(FUNC_NAME, \
    /* EXTRACTIONS */ \
    UPDATE_EXT;, \
 \
    /* NEURON_PRE
     * Retrieve input sum */ \
    float sum = inputs[to_index];, \
 \
    /* WEIGHT_OP */ \
    UPDATE_CALC;, \
 \
    /* NEURON_POST
     * no_op */ \
    ; \
)

#endif
