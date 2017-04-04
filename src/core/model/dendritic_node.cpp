#include "model/dendritic_node.h"

int DendriticNode::count = 0;

/* Constructor for an internal node */
DendriticNode::DendriticNode(int register_index, Layer *to_layer)
        : register_index(register_index),
          to_layer(to_layer),
          id(count++),
          conn(nullptr),
          second_order(false) { }

/* Constructor for a leaf node */
DendriticNode::DendriticNode(int register_index,
    Layer *to_layer, Connection *conn)
        : register_index(register_index),
          to_layer(to_layer),
          id(count++),
          conn(conn),
          second_order(false) { }

DendriticNode::~DendriticNode() {
    for (auto& child : children) delete child;
}

void DendriticNode::set_second_order() {
    // Check if leaf node
    if (is_leaf())
        ErrorManager::get_instance()->log_error(
            "Leaf dendritic nodes cannot be second order!");

    // This must be set before children are added
    if (children.size() > 1)
        ErrorManager::get_instance()->log_error(
            "Dendritic nodes must be set to second order before adding children!");

    second_order = true;
}

int DendriticNode::get_second_order_size() const {
    if (not is_second_order())
        ErrorManager::get_instance()->log_error(
            "Requested second order size on non-second order node!");
    else if (children.size() == 0)
        return 0;
    else
        return children[0]->conn->get_num_weights();
}

DendriticNode* DendriticNode::add_child() {
    if (is_leaf())
        ErrorManager::get_instance()->log_error(
            "Dendritic node cannot have children if it has a connection!");
    else if (is_second_order())
        ErrorManager::get_instance()->log_error(
            "Second order dendritic nodes cannot have internal children!");

    int child_register = this->register_index;
    if (children.size() > 0) ++child_register;
    auto child = new DendriticNode(child_register, this->to_layer);
    children.push_back(child);
    return child;
}

DendriticNode* DendriticNode::add_child(Connection *conn) {
    if (is_leaf())
        ErrorManager::get_instance()->log_error(
            "Dendritic node cannot have children if it has a connection!");

    // If this is not the first child, ensure sizes match
    if (is_second_order() and children.size() > 0
        and conn->get_num_weights() != children[0]->conn->get_num_weights())
        ErrorManager::get_instance()->log_error(
            "Second order connections must have identical sizes!");

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
