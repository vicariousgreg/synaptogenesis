#include "engine/kernel/aggregator.h"
#include "util/parallel.h"
#include "util/logger.h"
#include "util/resource_manager.h"
#include "util/tools.h"

/* Synaptic operations
 * |prior| is the current state of the neuron.
 * |input| is the synaptic input accomulated from one connection.
 *
 * ADD represent traditional excitatory input
 * SUB represent traditional inhibitory input
 * MULT and DIV represent modulatory input that can be used for gating
 * POOL represents max pooling
 * */
HOST float aggregate_add_host(float prior, float input)
    { return prior + input; }
HOST float aggregate_sub_host(float prior, float input)
    { return prior - input; }
HOST float aggregate_mult_host(float prior, float input)
    { return prior * input; }
HOST float aggregate_div_host(float prior, float input)
    { return prior / input; }
HOST float aggregate_pool_host(float prior, float input)
    { return MAX(prior, input); }
HOST float aggregate_add_heaviside_host(float prior, float input)
    { return prior + (input > 0); }
HOST float aggregate_sub_heaviside_host(float prior, float input)
    { return prior - (input > 0); }
HOST float aggregate_mult_heaviside_host(float prior, float input)
    { return prior * (input > 0); }

DEVICE float aggregate_add_device(float prior, float input)
    { return prior + input; }
DEVICE float aggregate_sub_device(float prior, float input)
    { return prior - input; }
DEVICE float aggregate_mult_device(float prior, float input)
    { return prior * input; }
DEVICE float aggregate_div_device(float prior, float input)
    { return prior / input; }
DEVICE float aggregate_pool_device(float prior, float input)
    { return MAX(prior, input); }
DEVICE float aggregate_add_heaviside_device(float prior, float input)
    { return prior + (input > 0); }
DEVICE float aggregate_sub_heaviside_device(float prior, float input)
    { return prior - (input > 0); }
DEVICE float aggregate_mult_heaviside_device(float prior, float input)
    { return prior * (input > 0); }

// Device pointers for memcpyFromSymbol
DEVICE AGGREGATOR ag_add  = aggregate_add_device;
DEVICE AGGREGATOR ag_sub  = aggregate_sub_device;
DEVICE AGGREGATOR ag_mult = aggregate_mult_device;
DEVICE AGGREGATOR ag_div  = aggregate_div_device;
DEVICE AGGREGATOR ag_pool = aggregate_pool_device;
DEVICE AGGREGATOR ag_add_heaviside  = aggregate_add_heaviside_device;
DEVICE AGGREGATOR ag_sub_heaviside  = aggregate_sub_heaviside_device;
DEVICE AGGREGATOR ag_mult_heaviside  = aggregate_mult_heaviside_device;

AGGREGATOR get_aggregator(Opcode opcode, DeviceID device_id) {
    AGGREGATOR x;
    if (ResourceManager::get_instance()->is_host(device_id))
        switch (opcode) {
            case ADD:  x = aggregate_add_host;   break;
            case SUB:  x = aggregate_sub_host;   break;
            case MULT: x = aggregate_mult_host;  break;
            case DIV:  x = aggregate_div_host;   break;
            case POOL: x = aggregate_pool_host;  break;
            case ADD_HEAVISIDE: x = aggregate_add_heaviside_host;  break;
            case SUB_HEAVISIDE: x = aggregate_sub_heaviside_host;  break;
            case MULT_HEAVISIDE: x = aggregate_mult_heaviside_host;  break;
        }
    else
#ifdef __CUDACC__
        switch (opcode) {
            case ADD:  cudaMemcpyFromSymbol(&x, ag_add, sizeof(void *));  break;
            case SUB:  cudaMemcpyFromSymbol(&x, ag_sub, sizeof(void *));  break;
            case MULT: cudaMemcpyFromSymbol(&x, ag_mult, sizeof(void *)); break;
            case DIV:  cudaMemcpyFromSymbol(&x, ag_div, sizeof(void *));  break;
            case POOL: cudaMemcpyFromSymbol(&x, ag_pool, sizeof(void *)); break;
            case ADD_HEAVISIDE: cudaMemcpyFromSymbol(&x, ag_add_heaviside, sizeof(void *)); break;
            case SUB_HEAVISIDE: cudaMemcpyFromSymbol(&x, ag_sub_heaviside, sizeof(void *)); break;
	    case MULT_HEAVISIDE: cudaMemcpyFromSymbol(&x, ag_mult_heaviside, sizeof(void *)); break;
        }
#else
        LOG_ERROR(
            "Tried to retrieve device aggregator in serial build!");
#endif
    return x;
}
