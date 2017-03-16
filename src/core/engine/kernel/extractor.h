#ifndef extractor_h
#define extractor_h

#include "util/constants.h"
#include "util/resource_manager.h"

/* Extractors are responsible for extracting values from output */
typedef float(*EXTRACTOR)(Output&, int delay);

void get_extractor(EXTRACTOR *dest, OutputType output_type, DeviceID device_id);

#endif
