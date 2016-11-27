#ifndef kernel_h
#define kernel_h

#include "util/parallel.h"
#include "util/constants.h"

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
    return 0.0;
}

class ConnectionData;

/* Clears input data */
GLOBAL void clear_data(float* data, int count);

/* Extractors are responsible for extracting values from output */
typedef float(*EXTRACTOR)(ConnectionData&, Output&);
void get_extractor(EXTRACTOR *dest, OutputType output_type);
DEVICE float extract_float(ConnectionData &conn_data, Output &out);
DEVICE float extract_int(ConnectionData &conn_data, Output &out);
DEVICE float extract_bit(ConnectionData &conn_data, Output &out);

/* Activators are responsible for performing connection computation */
typedef void(*ACTIVATOR)(ConnectionData);
void get_activator(ACTIVATOR *dest, ConnectionType conn_type);
GLOBAL void calc_fully_connected(ConnectionData conn_data);
GLOBAL void calc_one_to_one(ConnectionData conn_data);
GLOBAL void calc_convergent(ConnectionData conn_data);

/* Updaters are responsible for updating connection weights */
typedef void(*UPDATER)(ConnectionData);
void get_updater(UPDATER *dest, ConnectionType conn_type);
GLOBAL void update_fully_connected(ConnectionData conn_data);
GLOBAL void update_one_to_one(ConnectionData conn_data);
GLOBAL void update_convergent(ConnectionData conn_data);

#endif
