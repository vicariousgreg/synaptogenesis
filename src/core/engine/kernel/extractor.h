#ifndef extractor_h
#define extractor_h

#include "util/constants.h"

/* Extractors are responsible for extracting values from output */
typedef float(*EXTRACTOR)(Output&, int delay);

EXTRACTOR get_extractor(OutputType output_type, DeviceID device_id);

#endif
