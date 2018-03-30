#ifndef aggregator_h
#define aggregator_h

#include "util/constants.h"

/* Aggregators are responsible for aggregating input values based on opcode */
typedef float(*AGGREGATOR)(float prior, float input);

AGGREGATOR get_aggregator(Opcode opcode, DeviceID device_id);

#endif
