#ifndef tools_h
#define tools_h

#include <climits>
#include <random>

// Random number generator (not thread safe!)
static std::mt19937 generator(std::random_device{}());

#ifdef _OPENMP

// Create vector of generators for OpenMP threads
// This will be initialized in init_openmp_rand() in util/parallel.h
#include <omp.h>
#include <vector>
#define THREAD_SAFE_GENERATOR generators[omp_get_thread_num()]
static std::vector<std::mt19937> generators;

#else

// If OpenMP is not included, use the default generator
#define THREAD_SAFE_GENERATOR generator

#endif

// Float array setter
float fSet(float* arr, int size, float val, float fraction=1.0);

// Clears float array
void fClear(float* arr, int size);

/* Float and integer random generators
 *   Defaults
 *   0 - Max
 *   Min - Max
 */
inline float fRand()
    { return std::uniform_real_distribution<float>(0.0, 1.0)(generator); }
inline float fRand(float fMax)
    { return std::uniform_real_distribution<float>(0.0, fMax)(generator); }
inline float fRand(float fMin, float fMax)
    { return std::uniform_real_distribution<float>(fMin, fMax)(generator); }
inline int iRand()
    { return std::uniform_int_distribution<int>(0,INT_MAX)(generator); }
inline int iRand(int iMax)
    { return std::uniform_int_distribution<int>(0,iMax)(generator); }
inline int iRand(int iMin, int iMax)
    { return std::uniform_int_distribution<int>(iMin,iMax)(generator); }

/* Flat and integer array random generators */
void fRand(float* arr, int size, float fMin, float fMax, float fraction=1.0);
void iRand(int* arr, int size, int iMin, int iMax, float fraction=1.0);

/* Alternate randomizers */
void fRand_gaussian(float* arr, int size,
    float mean, float std_dev, float max, float fraction=1.0);
void fRand_lognormal(float* arr, int size,
    float mean, float std_dev, float max, float fraction=1.0);
void fRand_powerlaw(float* arr, int size,
    float exponent, float min, float max, float fraction=1.0);

/* Clears the diagonal of a weight matrix */
void clear_diagonal(float *mat, int dim);

#endif
