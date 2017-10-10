#include "engine/kernel/extractor.h"
#include "util/parallel.h"
#include "util/error_manager.h"
#include "util/resource_manager.h"

HOST float extract_float_host(Output &out, int delay) { return out.f; }
HOST float extract_int_host(Output &out, int delay) { return out.i; }
HOST float extract_bit_host(Output &out, int delay) {
    // delay & 0x1F    is equivalent to    delay % 32
    // The word offset is predetermined, so the remainder
    //   determines how far into the word to go to extract
    //   the relevant bit
    return (out.i << (delay & 0x1F)) >> 31;
}

DEVICE float extract_float_device(Output &out, int delay) { return out.f; }
DEVICE float extract_int_device(Output &out, int delay) { return out.i; }
DEVICE float extract_bit_device(Output &out, int delay) {
    // delay & 0x1F    is equivalent to    delay % 32
    // The word offset is predetermined, so the remainder
    //   determines how far into the word to go to extract
    //   the relevant bit
    return (out.i << (delay & 0x1F)) >> 31;
}

// Device pointers for memcpyFromSymbol
DEVICE EXTRACTOR x_float = extract_float_device;
DEVICE EXTRACTOR x_int = extract_int_device;
DEVICE EXTRACTOR x_bit = extract_bit_device;

EXTRACTOR get_extractor(OutputType output_type, DeviceID device_id) {
    EXTRACTOR x;
    if (ResourceManager::get_instance()->is_host(device_id))
        switch (output_type) {
            case FLOAT: x = extract_float_host;  break;
            case INT:   x = extract_int_host;    break;
            case BIT:   x = extract_bit_host;    break;
        }
    else
#ifdef __CUDACC__
        switch (output_type) {
            case FLOAT: cudaMemcpyFromSymbol(&x, x_float, sizeof(void *)); break;
            case INT:   cudaMemcpyFromSymbol(&x, x_int, sizeof(void *));   break;
            case BIT:   cudaMemcpyFromSymbol(&x, x_bit, sizeof(void *));   break;
        }
#else
        LOG_ERROR(
            "Tried to retrieve device extractor in serial build!");
#endif
    return x;
}
