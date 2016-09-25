#ifndef parallel_h
#define parallel_h

#ifdef PARALLEL

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "assert.h"

void cudaCheckError(const char* msg);

void cudaSync();

/** Checks cuda memory usage and availability */
void check_memory();

#endif

#endif
