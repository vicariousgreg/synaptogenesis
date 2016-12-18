#include "engine/kernel/extractor.h"

DEVICE float extract_float(int delay, Output &out) { return out.f; }
DEVICE float extract_int(int delay, Output &out) { return out.i; }
DEVICE float extract_bit(int delay, Output &out) {
    return (out.i >> (delay % 32)) & 1;
}

// Device pointers for memcpyFromSymbol
DEVICE EXTRACTOR x_float = extract_float;
DEVICE EXTRACTOR x_int = extract_int;
DEVICE EXTRACTOR x_bit = extract_bit;

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
