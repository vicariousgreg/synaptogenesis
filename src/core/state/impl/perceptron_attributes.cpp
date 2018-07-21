#include <string>

#include "state/impl/perceptron_attributes.h"
#include "engine/kernel/synapse_kernel.h"
#include "util/logger.h"

REGISTER_ATTRIBUTES(PerceptronAttributes, "perceptron", FLOAT)

/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(PerceptronAttributes, perceptron_attribute_kernel,
    float *f_outputs = (float*)outputs;
    int num_weights = attribute_data.num_weights / size;

    ,

    float input = inputs[nid];
    SHIFT_FLOAT_OUTPUTS(f_outputs, input / num_weights);
)

/******************************************************************************/
/************************** PERCEPTRON KERNELS ********************************/
/******************************************************************************/

CALC_ALL(activate_perceptron,
    ,
    float sum = 0.0;
    ,
    sum += (extract(outputs[from_index], delay) > 0) * weights[weight_index];
    ,
    inputs[to_index] = aggregate(inputs[to_index], sum);
; );

KernelList<SYNAPSE_ARGS> PerceptronAttributes::get_activators(Connection *conn) {
    if (conn->second_order or conn->convolutional)
        LOG_ERROR(
            "Unimplemented connection type!");

    try {
        return { activate_perceptron_map.at(conn->get_type()) };
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Unimplemented connection type!");
    }
}

#define LEARNING_RATE 0.05

CALC_ALL(update_perceptron,
    float* expected_output =
        ((PerceptronAttributes*)synapse_data.attributes)->expected_output.get();
    ,
    float delta = expected_output[to_index] - destination_outputs[to_index].f;
    ,
    weights[weight_index] += LEARNING_RATE * delta * outputs[from_index].f;
    ,
; );

KernelList<SYNAPSE_ARGS> PerceptronAttributes::get_updaters(Connection *conn) {
    if (conn->second_order or conn->convolutional)
        LOG_ERROR(
            "Unimplemented connection type!");

    try {
        return { update_perceptron_map.at(conn->get_type()) };
    } catch (std::out_of_range) { }

    LOG_ERROR(
        "Unimplemented connection type!");
}

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

PerceptronAttributes::PerceptronAttributes(Layer *layer)
        : Attributes(layer, FLOAT) {
    this->expected_output = Attributes::create_neuron_variable<float>();
    Attributes::register_neuron_variable("expected", &expected_output);
}
