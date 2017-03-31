#include "model/dendritic_node.h"

/* Constructor for an internal node */
DendriticNode::DendriticNode(int register_index, Layer *to_layer)
        : register_index(register_index),
          to_layer(to_layer),
          conn(nullptr) { }

/* Constructor for a leaf node */
DendriticNode::DendriticNode(int register_index,
    Layer *to_layer, Connection *conn)
        : register_index(register_index),
          to_layer(to_layer),
          conn(conn) { }

DendriticNode::~DendriticNode() {
    for (auto& child : children) delete child;
}

DendriticNode* DendriticNode::add_child() {
    if (this->conn != nullptr)
        ErrorManager::get_instance()->log_error(
            "Dendritic node cannot have children if it has a connection!");
    int child_register = this->register_index;
    if (children.size() > 0) ++child_register;
    auto child = new DendriticNode(child_register, this->to_layer);
    children.push_back(child);
    return child;
}

DendriticNode* DendriticNode::add_child(Connection *conn) {
    if (this->conn != nullptr)
        ErrorManager::get_instance()->log_error(
            "Dendritic node cannot have children if it has a connection!");
    auto child = new DendriticNode(this->register_index, this->to_layer, conn);
    children.push_back(child);
    return child;
}

int DendriticNode::get_max_register_index() const {
    int max_register = this->register_index;
    for (auto& child : children) {
        int child_register = child->get_max_register_index();
        if (child_register > max_register)
            max_register = child_register;
    }
    return max_register;
}

const DendriticNodeList DendriticNode::get_children() const {
    return children;
}
