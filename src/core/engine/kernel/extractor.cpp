#include "engine/kernel/extractor.h"

DEVICE float extract_float(Output &out, int delay) { return out.f; }
DEVICE float extract_int(Output &out, int delay) { return out.i; }
DEVICE float extract_bit(Output &out, int delay) {
    // delay & 0x1F    is equivalent to    delay % 32
    // The word offset is predetermined, so the remainder
    //   determines how far into the word to go to extract
    //   the relevant bit
    return (out.i << (delay & 0x1F)) >> 31;
}

// Device pointers for memcpyFromSymbol
DEVICE EXTRACTOR x_float = extract_float;
DEVICE EXTRACTOR x_int = extract_int;
DEVICE EXTRACTOR x_bit = extract_bit;

void get_extractor(EXTRACTOR *dest, OutputType output_type) {
#ifdef __CUDACC__
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
