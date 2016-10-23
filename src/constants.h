#ifndef constants_h
#define constants_h

#include "parallel.h"

/* Size (in Outputs) of output history.
 * A size of 1 indicates the usage of 1 Output, which on most systems is
 *     4 bytes, or 32 bits.  This means that a history of 32 timesteps will
 *     be kept for neuron spikes, as they are represented by a single bit.
 *     For rate encoding, one Output is used for timestep.  Much more space
 *     is needed to hold longer delays in such a network.
 * The history size limits the longest connection delay possible, and also
 *     affects learning for STDP rules that use more than just the most
 *     recent Output.
 */
#define HISTORY_SIZE 1

/* Output type enumeration.
 * Different drivers may use different output formats.
 * This enumeration is used to keep track of this information.
 */
enum OutputType {
    FLOAT,
    BIT,
    INT
};

/* Output union.
 * This is used to avoid unnecesary templating.
 * Because all outputs will be of the same type, it would be a waste of
 *     space to hold an enumeration in each union instance.
 * Instead, objects should know what type the output is for all instances.
 */
union Output {
    float f;
    int i;
};

/* IO type enumeration.
 * Indicates sensorimotor connectivity.
 * Input layers receive sensory input from the environment.
 * Output layers send motor output to the environment.
 * Input/Output layers do both.
 * Internal layers do neither.
 */
enum IOType {
    INPUT,
    INPUT_OUTPUT,
    OUTPUT,
    INTERNAL,
    IO_TYPE_SIZE
};

/* Matrix Type enumeration.
 * Fully connected represents an n x m matrix.
 * One-to-one represents an n size vector connecting two layers
 *   of idential sizes.
 * Convergent connections converge from a larger layer to a smaller one.
 * Divergent connections diverge from a larger layer to a smaller one.
 * Convolutional layers share weights.
 */
enum ConnectionType {
    FULLY_CONNECTED,
    ONE_TO_ONE,
    DIVERGENT,
    DIVERGENT_CONVOLUTIONAL,
    CONVERGENT,
    CONVERGENT_CONVOLUTIONAL
};

/* Synaptic operation opcode
 * Defines how activity across a connection interacts with the current state.
 * This allows for more complex synaptic functions.
 * */
enum Opcode {
    ADD,
    SUB,
    MULT,
    DIV
};


/* Synaptic operations
 * |prior| is the current state of the neuron.
 * |input| is the synaptic input accomulated from one connection.
 *
 * ADD represent traditional excitatory input
 * SUB represent traditional inhibitory input
 * MULT and DIV represent modulatory input that can be used for gating
 * */
inline DEVICE float calc(Opcode opcode, float prior, float input) {
    switch (opcode) {
        case ADD:  return prior + input;
        case SUB:  return prior - input;
        case MULT: return prior * (1+input);
        case DIV:  return prior / (1+input);
    }
    #ifdef PARALLEL assert(false);
    #else throw "Unrecognized connection operation!";
    #endif
    return 0.0;
}

#endif
