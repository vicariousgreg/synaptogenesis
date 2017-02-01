#ifndef dendritic_node_h
#define dendritic_node_h

#include <vector>

#include "model/connection.h"
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
        /* Constructor for an internal node */
        DendriticNode(int register_index, Layer *to_layer)
                : register_index(register_index),
                  to_layer(to_layer),
                  conn(NULL) { }

        /* Constructor for a leaf node */
        DendriticNode(int register_index, Layer *to_layer, Connection *conn)
                : register_index(register_index),
                  to_layer(to_layer),
                  conn(conn) { }

        /* Add a child internal node */
        DendriticNode *add_child();
        /* Add a child leaf node */
        DendriticNode *add_child(Connection *conn);

        /* Constant getters */
        int get_max_register_index() const;
        const DendriticNodeList get_children() const { return children; }

        Connection* const conn;
        Layer* const to_layer;
        const int register_index;

    private:
        DendriticNodeList children;
};

#endif
