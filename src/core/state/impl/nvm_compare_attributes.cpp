#include "state/impl/nvm_compare_attributes.h"
#include "state/weight_matrix.h"
#include "engine/kernel/synapse_kernel.h"

#define GATE_THRESHOLD 0.5f

REGISTER_ATTRIBUTES(NVMCompareAttributes, "nvm_compare", FLOAT)
USE_WEIGHT_MATRIX(NVMWeightMatrix, "nvm_compare")

NVMCompareAttributes::NVMCompareAttributes(Layer *layer)
        : NVMHeavisideAttributes(layer) {
    this->true_state = Attributes::create_neuron_variable<float>(0.0);
    Attributes::register_neuron_variable("true_state", &true_state);

    this->tolerance = 1. - std::stof(
        layer->get_parameter("tolerance", "1.0"));
}

/******************************************************************************/
/***************************** ACTIVATOR KERNELS ******************************/
/******************************************************************************/

#define EXTRACT \
    NVMCompareAttributes *nvm_att = (NVMCompareAttributes*)synapse_data.attributes;

#define INIT_SUM \
    float sum = 0.0;

#define WEIGHT_OP \
    sum += extract(outputs[from_index], delay) * weights[weight_index]; \

#define AGG_SUM_WEIGHTED(weight) \
    inputs[to_index] = weight * aggregate(inputs[to_index], sum);

CALC_ALL_DUAL(activate_nvm_compare_plastic,
    NVMCompareAttributes *nvm_att = (NVMCompareAttributes*)synapse_data.attributes;
    NVMWeightMatrix *nvm_mat = (NVMWeightMatrix*)synapse_data.matrix;

    float ag = nvm_att->activity_gate;
    float lg = nvm_att->learning_gate;
    if (ag < GATE_THRESHOLD and lg < GATE_THRESHOLD) return;

    float* true_state = nvm_att->true_state.get();
    float norm = nvm_mat->norm;
    float tolerance = nvm_att->tolerance;
    ,
        INIT_SUM
        ,
            WEIGHT_OP
        ,
        if (ag > GATE_THRESHOLD) {
            sum -= true_state[to_index] *
                ((tolerance * num_weights_per_neuron) - 1);
            AGG_SUM_WEIGHTED(ag)
        }
        if (lg > GATE_THRESHOLD) {
            float temp = inputs[to_index] = true_state[to_index];
            ,
                weights[weight_index] =
                    extract(outputs[from_index], delay) *
                    temp / norm;
            ,
        }
);


KernelList<SYNAPSE_ARGS> NVMCompareAttributes::get_activators(Connection *conn) {
    try {
        if (conn->plastic)
            return { activate_nvm_compare_plastic_map.at(conn->get_type()) };
        else return NVMHeavisideAttributes::get_activators(conn);
    } catch(std::out_of_range) { }

    // Log an error if the connection type is unimplemented
    LOG_ERROR("Unimplemented connection type!");
}
