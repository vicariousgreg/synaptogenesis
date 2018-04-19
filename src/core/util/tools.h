#ifndef tools_h
#define tools_h

#include <climits>
#include <random>

// Random number generator
static std::default_random_engine generator(std::random_device{}());

// Float array setter
float fSet(float* arr, int size, float val, float fraction=1.0);

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

#endif
