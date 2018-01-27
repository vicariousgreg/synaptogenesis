#ifndef aggregator_h
#define aggregator_h

#include <cmath>

// Different min, max, and assert functions are used on the host and device
#ifdef __CUDACC__
#define MIN min
#define MAX max
#else
#include <algorithm>
#include <assert.h>
#define MIN std::fmin
#define MAX std::fmax
#endif

#include "util/constants.h"

/* Aggregators are responsible for aggregating input values based on opcode */
typedef float(*AGGREGATOR)(float prior, float input);

AGGREGATOR get_aggregator(Opcode opcode, DeviceID device_id);

#endif
