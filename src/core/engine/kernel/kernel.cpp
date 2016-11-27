#include <cmath>
#include "engine/kernel/kernel.h"
#include "engine/instruction.h"
#include "util/parallel.h"
#include "util/error_manager.h"

/******************************************************************************/
/************************** OUTPUT EXTRACTORS *********************************/
/******************************************************************************/

// Device pointers for memcpyFromSymbol
DEVICE EXTRACTOR x_float = extract_float;
DEVICE EXTRACTOR x_int = extract_int;
DEVICE EXTRACTOR x_bit = extract_bit;

DEVICE float extract_float(ConnectionData &conn_data, Output &out) { return out.f; }
DEVICE float extract_int(ConnectionData &conn_data, Output &out) { return out.i; }
DEVICE float extract_bit(ConnectionData &conn_data, Output &out) {
    return (out.i >> (conn_data.delay % 32)) & 1;
}

void get_extractor(EXTRACTOR *dest, OutputType output_type) {
#ifdef PARALLEL
    switch (output_type) {
        case FLOAT:
            cudaMemcpyFromSymbol(dest, x_float, sizeof(void *));
            break;
        case INT:
            cudaMemcpyFromSymbol(dest, x_int, sizeof(void *));
            break;
        case BIT:
            cudaMemcpyFromSymbol(dest, x_bit, sizeof(void *));
            break;
    }
#else
    switch (output_type) {
        case FLOAT:
            *dest = extract_float;
            break;
        case INT:
            *dest = extract_int;
            break;
        case BIT:
            *dest = extract_bit;
            break;
    }
#endif
}

/******************************************************************************/
/**************************** DATA CLEARING ***********************************/
/******************************************************************************/

GLOBAL void clear_data(float* data, int count) {
#ifdef PARALLEL
    int nid = blockIdx.x * blockDim.x + threadIdx.x;
    if (nid < count)
#else
    for (int nid = 0; nid < count; ++nid)
#endif
        data[nid] = 0.0;
}
