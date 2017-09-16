#include "engine/kernel/attribute_data.h"
#include "state/state.h"
#include "state/attributes.h"

AttributeData::AttributeData(Layer *layer, State *state) :
        attributes(state->get_attributes_pointer(layer)),
        input(state->get_input(layer)),
        output(state->get_output(layer)),
        expected(state->get_expected(layer)),
        layer_index(state->get_layer_index(layer)),
        other_start_index(state->get_other_start_index(layer)),
        size(layer->size),
        num_weights(layer->get_num_weights()),
        plastic(layer->plastic) {
    // Calculate history size
    auto output_type = state->get_output_type(layer);
    int max_delay_registers = 0;
    for (auto& conn : layer->get_output_connections()) {
        int delay_registers = get_word_index(conn->delay, output_type);
        if (delay_registers > max_delay_registers)
            max_delay_registers = delay_registers;
    }
    history_size = max_delay_registers + 1;
}
