#ifndef constants_h
#define constants_h

#include <map>
#include "util/error_manager.h"

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
    DIVERGENT,
    SPARSE        // Not added to maps because it's an internal type
} ConnectionType;

static std::map<ConnectionType, std::string> ConnectionTypeStrings = {
    {FULLY_CONNECTED, "fully connected"},
    {SUBSET, "subset"},
    {ONE_TO_ONE, "one to one"},
    {CONVERGENT, "convergent"},
    {DIVERGENT, "divergent"},
};

static std::map<std::string, ConnectionType> ConnectionTypes {
    {"fully connected", FULLY_CONNECTED},
    {"subset", SUBSET},
    {"one to one", ONE_TO_ONE},
    {"convergent", CONVERGENT},
    {"divergent", DIVERGENT},
};

inline ConnectionType get_connection_type(std::string name) {
    try {
        return ConnectionTypes.at(name);
    } catch (...) {
        LOG_ERROR(
            "Unrecognized ConnectionType: " + name);
    }
}

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
    MODULATE,
    GAP,
    ADD_HEAVISIDE,
    SUB_HEAVISIDE,
    MULT_HEAVISIDE,
} Opcode;

static std::map<Opcode, std::string> OpcodeStrings = {
    {ADD, "add"},
    {SUB, "sub"},
    {MULT, "mult"},
    {DIV, "div"},
    {POOL, "pool"},
    {REWARD, "reward"},
    {MODULATE, "modulate"},
    {GAP, "gap"},
    {ADD_HEAVISIDE, "add_heaviside"},
    {SUB_HEAVISIDE, "sub_heaviside"},
    {MULT_HEAVISIDE, "mult_heaviside"},
};

static std::map<std::string, Opcode> Opcodes = {
    {"add", ADD},
    {"sub", SUB},
    {"mult", MULT},
    {"div", DIV},
    {"pool", POOL},
    {"reward", REWARD},
    {"modulate", MODULATE},
    {"gap", GAP},
    {"add_heaviside", ADD_HEAVISIDE},
    {"sub_heaviside", SUB_HEAVISIDE},
    {"mult_heaviside", MULT_HEAVISIDE},
};

inline Opcode get_opcode(std::string name) {
    try {
        return Opcodes.at(name);
    } catch (...) {
        LOG_ERROR(
            "Unrecognized Opcode: " + name);
    }
}

/* Enumeration of cluster types. */
typedef enum {
    PARALLEL,
    SEQUENTIAL,
    FEEDFORWARD
} ClusterType;

static std::map<ClusterType, std::string> ClusterTypeStrings = {
    {PARALLEL, "parallel"},
    {SEQUENTIAL, "sequential"},
    {FEEDFORWARD, "feedforward"}
};

static std::map<std::string, ClusterType> ClusterTypes = {
    {"parallel", PARALLEL},
    {"sequential", SEQUENTIAL},
    {"feedforward", FEEDFORWARD},
};

inline ClusterType get_cluster_type(std::string name) {
    try {
        return ClusterTypes.at(name);
    } catch (...) {
        LOG_ERROR(
            "Unrecognized ClusterType: " + name);
    }
}

/* Typedef for Device Identifier */
typedef unsigned int DeviceID;

#endif
