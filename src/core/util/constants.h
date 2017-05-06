#ifndef constants_h
#define constants_h

/* Output type enumeration.
 * Different engines may use different output formats.
 * This enumeration is used to keep track of this information.
 */
typedef enum OutputType {
    FLOAT,
    BIT,
    INT
} OutputType;

static OutputType OutputTypes[] = { FLOAT, BIT, INT };

/* Returns the number of timesteps represented by one Output value */
inline int get_timesteps_per_output(OutputType output_type) {
    switch (output_type) {
        case FLOAT:
        case INT:
            return 1;
        case BIT:
            return sizeof(unsigned int) * 8;
    }
}

/* Gets a word index given a delay and an output type */
inline int get_word_index(int delay, OutputType output_type) {
    return delay / get_timesteps_per_output(output_type);
}

/* Output union.
 * This is used to avoid unnecesary templating.
 * Because all outputs will be of the same type, it would be a waste of
 *     space to hold an enumeration in each union instance.
 * Instead, objects should know what type the output is for all instances.
 */
union Output {
    float f;
    unsigned int i;
};

/* IO type enumeration.
 * Indicates sensorimotor connectivity.
 * Input layers receive sensory input from the environment.
 * Output layers send motor output to the environment.
 * Input/Output layers do both.
 * Internal layers do neither.
 */
typedef enum {
    INPUT = 1,
    OUTPUT = 2,
    EXPECTED = 4,
} IOType;

typedef unsigned char IOTypeMask;

/* Matrix Type enumeration.
 * Fully connected represents an n x m matrix.
 * One-to-one represents an n size vector connecting two layers
 *   of idential sizes.
 * Convergent connections converge from a larger layer to a smaller one.
 * Convolutional layers share weights.
 */
typedef enum {
    FULLY_CONNECTED,
    SUBSET,
    ONE_TO_ONE,
    CONVERGENT,
    CONVOLUTIONAL,
    DIVERGENT
} ConnectionType;

/* Synaptic operation opcode.
 * Defines how activity across a connection interacts with the current state.
 * This allows for more complex synaptic functions.
 * */
typedef enum {
    ADD,
    SUB,
    MULT,
    DIV,
    POOL,
    REWARD,
    MODULATE
} Opcode;

/* Enumeration of cluster types. */
typedef enum {
    PARALLEL,
    SEQUENTIAL,
    FEEDFORWARD
} ClusterType;

/* Typedef for Device Identifier */
typedef unsigned int DeviceID;

#endif
