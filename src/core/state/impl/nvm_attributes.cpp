#include <math.h>
#include <string.h>

#include "state/impl/nvm_attributes.h"
#include "state/weight_matrix.h"
#include "engine/kernel/synapse_kernel.h"

REGISTER_ATTRIBUTES(NVMAttributes, "nvm", FLOAT)
REGISTER_WEIGHT_MATRIX(NVMWeightMatrix, "nvm")

/******************************************************************************/
/************************** CLASS FUNCTIONS ***********************************/
/******************************************************************************/

NVMAttributes::NVMAttributes(Layer *layer)
        : Attributes(layer, FLOAT) {
    this->state = Attributes::create_neuron_variable<float>(0.0);
    Attributes::register_neuron_variable("state", &state);
}

void NVMAttributes::process_weight_matrix(WeightMatrix* matrix) {
    NVMWeightMatrix *s_mat = (NVMWeightMatrix*)matrix;

    // Retrieve connection and matrix data pointer
    Connection *conn = matrix->connection;
    s_mat->norm = std::stof(conn->get_parameter("norm", "1.0"));
}


/******************************************************************************/
/******************************** KERNEL **************************************/
/******************************************************************************/

BUILD_ATTRIBUTE_KERNEL(NVMAttributes, nvm_kernel,
    float* state = att->state.get();
    float *f_outputs = (float*)outputs;
    ,
    float input = inputs[nid];
    state[nid] = input;
    SHIFT_FLOAT_OUTPUTS(f_outputs, tanh(input));
)

/******************************************************************************/
/***************************** ACTIVATOR KERNELS ******************************/
/******************************************************************************/

#define EXTRACT \
    NVMAttributes *nvm_att = (NVMAttributes*)synapse_data.attributes;

#define INIT_SUM \
    float sum = 0.0;

#define WEIGHT_OP \
    sum += extract(outputs[from_index], delay) * weights[weight_index]; \

#define AGG_SUM \
    inputs[to_index] = aggregate(inputs[to_index], sum);

CALC_SUBSET(activate_nvm_decay,
    NVMAttributes *nvm_att = (NVMAttributes*)synapse_data.attributes; \
    ,
    bool gv = false;
    ,
        gv = gv or not (extract(outputs[from_index], delay) > 0.0);
    ,
    nvm_att->activity_gate = gv;
);
CALC_SUBSET(activate_nvm_gate,
    NVMAttributes *nvm_att = (NVMAttributes*)synapse_data.attributes; \
    ,
    bool gv = false;
    ,
        gv = gv or (extract(outputs[from_index], delay) > 0.0);
    ,
    nvm_att->activity_gate = gv;
);
CALC_SUBSET(activate_nvm_learning,
    NVMAttributes *nvm_att = (NVMAttributes*)synapse_data.attributes; \
    ,
    bool gv = false;
    ,
        gv = gv or (extract(outputs[from_index], delay) > 0.0);
    ,
    nvm_att->learning_gate = gv;
);

CALC_ALL(activate_nvm,
    EXTRACT
    ,
    INIT_SUM
    ,
        WEIGHT_OP
    ,
    AGG_SUM
);

CALC_ALL(activate_nvm_gated,
    NVMAttributes *nvm_att = (NVMAttributes*)synapse_data.attributes;
    if (not nvm_att->activity_gate) return;
    ,
    INIT_SUM
    ,
        WEIGHT_OP
    ,
    AGG_SUM
);

CALC_ALL_DUAL(activate_nvm_plastic,
    NVMAttributes *nvm_att = (NVMAttributes*)synapse_data.attributes;
    NVMWeightMatrix *nvm_mat = (NVMWeightMatrix*)synapse_data.matrix;

    bool ag = nvm_att->activity_gate;
    bool lg = nvm_att->learning_gate;
    if (not (ag or lg)) return;

    float* state = nvm_att->state.get();
    float norm = nvm_mat->norm;
    ,
        INIT_SUM
        ,
            WEIGHT_OP
        ,
        if (ag) {
            AGG_SUM
        }
        if (lg) {
            float temp = state[to_index] - sum;
            ,
                weights[weight_index] +=
                    extract(outputs[from_index], delay) *
                    temp / norm;
            ,
        }
);


KernelList<SYNAPSE_ARGS> NVMAttributes::get_activators(Connection *conn) {
    try {
        if (not conn->second_order) {
            if (conn->get_parameter("decay", "false") == "true")
                return { get_activate_nvm_decay() } ;
            else if (conn->get_parameter("gate", "false") == "true")
                return { get_activate_nvm_gate() } ;
            else if (conn->get_parameter("learning", "false") == "true")
                return { get_activate_nvm_learning() } ;
            else if (conn->plastic)
                return { activate_nvm_plastic_map.at(conn->get_type()) };
            else if (conn->get_parameter("gated", "false") == "true")
                return { activate_nvm_gated_map.at(conn->get_type()) };
            else
                return { activate_nvm_map.at(conn->get_type()) };
        }
    } catch(std::out_of_range) { }

    // Log an error if the connection type is unimplemented
    LOG_ERROR("Unimplemented connection type!");
}
