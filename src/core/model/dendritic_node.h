#ifndef dendritic_node_h
#define dendritic_node_h

#include <vector>

#include "model/connection.h"
#include "util/constants.h"
#include "util/error_manager.h"

class DendriticNode {
    public:
        DendriticNode(int register_index, Layer *to_layer)
                : register_index(register_index),
                  to_layer(to_layer),
                  conn(NULL) { }

        DendriticNode(int register_index, Layer *to_layer, Connection *conn)
                : register_index(register_index),
                  to_layer(to_layer),
                  conn(conn) { }

        DendriticNode *add_child();
        DendriticNode *add_child(Connection *conn);
        int get_max_register_index();

        int register_index;
        std::vector<DendriticNode*> children;
        Connection *conn;
        Layer *to_layer;
};

#endif
