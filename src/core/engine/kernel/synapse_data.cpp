#include "engine/kernel/synapse_data.h"
#include "network/connection.h"
#include "network/dendritic_node.h"
#include "network/structure.h"
#include "state/state.h"
#include "state/attributes.h"

SynapseData::SynapseData(DendriticNode *parent_node,
    Connection *conn, State *state, bool updater) :
        attributes(state->get_attributes_pointer(conn->to_layer)),
        extractor(state->get_connection_extractor(conn)),
        aggregator(state->get_connection_aggregator(conn)),
        subset_config(conn->get_config()->get_subset_config()),
        arborized_config(conn->get_config()->get_arborized_config()),
        connection(*conn),
        from_layer(*conn->from_layer),
        to_layer(*conn->to_layer),
        matrix(state->get_matrix_pointer(conn)),
        weights(
            (conn->second_order_host)
                ? state->get_matrix(conn)->second_order_weights
                : state->get_matrix(conn)->weights),
        second_order_host_matrix(
            (conn->second_order_slave)
                ? state->get_matrix_pointer(
                    parent_node->get_second_order_connection())
                : nullptr),
        num_weights(conn->get_num_weights()),
        inputs(state->get_input(conn->to_layer, parent_node->register_index)),
        destination_outputs(state->get_output(conn->to_layer)) {
    // Updaters can only use outputs if they are within the same feedforward
    //   structure.  Otherwise, race conditions prevent outputs from being
    //   safely accessed.  Activators don't have this problem.
    bool get_outputs = not updater;
    if (updater) {
        auto from_structure = conn->from_layer->structure;
        auto to_structure = conn->to_layer->structure;
        get_outputs = (from_structure == to_structure
            and from_structure->cluster_type == FEEDFORWARD);
    }

    if (get_outputs) {
        auto output_type = Attributes::get_output_type(conn->from_layer);
        if (state->is_inter_device(conn))
            outputs = state->get_device_output_buffer(conn,
                get_word_index(conn->delay, output_type));
        else
            outputs = state->get_output(conn->from_layer,
                get_word_index(conn->delay, output_type));
    }
}

SynapseData::SynapseData(WeightMatrix *matrix, Connection *conn,
    Pointer<void*> p_to_p) :
        attributes(nullptr),
        extractor(nullptr),
        aggregator(nullptr),
        subset_config(conn->get_config()->get_subset_config()),
        arborized_config(conn->get_config()->get_arborized_config()),
        connection(*conn),
        from_layer(*conn->from_layer),
        to_layer(*conn->to_layer),
        matrix(matrix),
        second_order_host_matrix(nullptr),
        num_weights(conn->get_num_weights()),
        pointer_to_pointer(p_to_p) { }
