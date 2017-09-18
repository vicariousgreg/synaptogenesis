#ifndef dendritic_node_h
#define dendritic_node_h

#include <vector>
#include <string>

#include "network/connection.h"
#include "util/constants.h"
#include "util/error_manager.h"

class DendriticNode;
typedef std::vector<DendriticNode*> DendriticNodeList;

/* Represents a split point on a dendritic tree.
 *
 * Leaf nodes represent inputs from other neurons and have an associated
 *     connection. Internal nodes aggregate these inputs, allowing them
 *     to flow up the tree and into the destination neuron.
 *
 * Computations are performed as a depth first search through the dendritic
 *     tree.  Because intermediate results need to be stored for this to work,
 *     each node has an associated register index, indicating where to aggregate
 *     the final result.
 */
class DendriticNode {
    public:
        virtual ~DendriticNode();

        /* Sets this node as a second order node.
         * This can be applied to internal nodes with only leaf children.
         * It indicates that the synaptic activities of corresponding synapses
         *   should be multiplied together before aggregation.
         * If a node is second order, all of its children must have connections
         *   of the same size */
        void set_second_order();

        /* Returns whether this node is second order */
        bool is_second_order() const { return second_order; }

        /* Returns the size of the second order input buffer */
        int get_second_order_size() const;

        /* Returns the host connection for second order nodes */
        Connection* get_second_order_connection() const;

        /* Add a child internal node */
        DendriticNode *add_child(std::string name);

        /* Add a child leaf node */
        DendriticNode *add_child(Connection *conn);

        /* Returns whether this node is a leaf node */
        bool is_leaf() const
            { return not is_second_order() and conn != nullptr; }

        /* Constant getters */
        int get_max_register_index() const;
        const DendriticNodeList get_children() const;

        DendriticNode* const parent;
        Connection* const conn;
        Layer* const to_layer;
        const int register_index;
        const int id;
        const std::string name;

    private:
        friend class Layer;

        /* Constructor for a root node */
        DendriticNode(Layer *to_layer);

        /* Constructor for an internal node */
        DendriticNode(DendriticNode *parent, Layer *to_layer,
            int register_index, std::string name);

        /* Constructor for a leaf node */
        DendriticNode(DendriticNode *parent, Layer *to_layer,
            int register_index, Connection *conn);

        Connection* second_order_conn;
        DendriticNodeList children;
        bool second_order;
};

#endif
