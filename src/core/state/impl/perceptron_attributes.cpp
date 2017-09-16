#include <string>

#include "state/impl/perceptron_attributes.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/error_manager.h"

REGISTER_ATTRIBUTES(PerceptronAttributes, "perceptron")

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(PerceptronAttributes, perceptron_attribute_kernel,
    float *f_outputs = (float*)outputs;
    float *expecteds = (float*)attribute_data.expected.get();
    int num_weights = attribute_data.num_weights / size;

    ,

    float next_value = f_outputs[nid];
    int index;
    for (index = 0 ; index < history_size-1 ; ++index) {
        float curr_value = next_value;
        next_value = f_outputs[size * (index + 1) + nid];
        f_outputs[size * index + nid] = next_value;
    }
    float input = inputs[nid];
    f_outputs[size * index + nid] = input / num_weights;
)

/******************************************************************************/
/************************** PERCEPTRON KERNELS ********************************/
/******************************************************************************/

CALC_ALL(activate_perceptron,
    ,
    float sum = 0.0;
    ,
    sum += (extractor(outputs[from_index], delay) > 0) * weights[weight_index];
    ,
    inputs[to_index] = calc(opcode, inputs[to_index], sum);
; );

Kernel<SYNAPSE_ARGS> PerceptronAttributes::get_activator(Connection *conn) {
    if (conn->second_order)
        ErrorManager::get_instance()->log_error(
            "Unimplemented connection type!");

    std::map<ConnectionType, Kernel<SYNAPSE_ARGS> > funcs;
    funcs[FULLY_CONNECTED]  = get_activate_perceptron_fully_connected();
    funcs[SUBSET]           = get_activate_perceptron_subset();
    funcs[ONE_TO_ONE]       = get_activate_perceptron_one_to_one();
    funcs[CONVERGENT]       = get_activate_perceptron_convergent();
    funcs[DIVERGENT]        = get_activate_perceptron_divergent();

    try {
        return funcs.at(conn->type);
    } catch (std::out_of_range) {
        ErrorManager::get_instance()->log_error(
            "Unimplemented connection type!");
    }
}

#define LEARNING_RATE 0.05

CALC_ALL(update_perceptron,
    float* expecteds =
        (float*)(synapse_data.attributes
            ->expected.get(synapse_data.to_start_index));
    ,
    float delta = expecteds[to_index] - destination_outputs[to_index].f;
    ,
    weights[weight_index] += LEARNING_RATE * delta * outputs[from_index].f;
    ,
; );

Kernel<SYNAPSE_ARGS> PerceptronAttributes::get_updater(Connection *conn) {
    if (conn->second_order)
        ErrorManager::get_instance()->log_error(
            "Unimplemented connection type!");

    std::map<ConnectionType, Kernel<SYNAPSE_ARGS> > funcs;
    funcs[FULLY_CONNECTED]  = get_update_perceptron_fully_connected();
    funcs[SUBSET]           = get_update_perceptron_subset();
    funcs[ONE_TO_ONE]       = get_update_perceptron_one_to_one();
    funcs[CONVERGENT]       = get_update_perceptron_convergent();
    funcs[DIVERGENT]        = get_update_perceptron_divergent();

    try {
        return funcs.at(conn->type);
    } catch (std::out_of_range) {
        ErrorManager::get_instance()->log_error(
            "Unimplemented connection type!");
    }
}

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

PerceptronAttributes::PerceptronAttributes(LayerList &layers)
        : Attributes(layers, FLOAT) { }
