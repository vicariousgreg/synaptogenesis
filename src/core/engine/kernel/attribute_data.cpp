#include "engine/kernel/attribute_data.h"
#include "state/state.h"
#include "state/attributes.h"
#include "util/error_manager.h"

AttributeData::AttributeData(Layer *layer, State *state) :
        attributes(state->get_attributes_pointer(layer)),
        input(state->get_input(layer)),
        output(state->get_output(layer)),
        other_start_index(state->get_other_start_index(layer)),
        size(layer->size) {
    // Calculate history size
    int timesteps_per_output =
        get_timesteps_per_output(state->get_output_type(layer->structure)); 
    int max_delay_registers = 0;
    for (auto& conn : layer->get_output_connections()) {
        int delay_registers = conn->delay / timesteps_per_output;
        if (delay_registers > max_delay_registers)
            max_delay_registers = delay_registers;
    } 
    history_size = max_delay_registers + 1;
}
