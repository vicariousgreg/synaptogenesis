#include "model/dendritic_node.h"
#include "model/layer.h"

int DendriticNode::count = 0;

/* Constructor for an internal node */
DendriticNode::DendriticNode(int register_index, Layer *to_layer)
        : register_index(register_index),
          to_layer(to_layer),
          id(count++),
          conn(nullptr),
          second_order_conn(nullptr),
          second_order(false) { }

/* Constructor for a leaf node */
DendriticNode::DendriticNode(int register_index,
    Layer *to_layer, Connection *conn)
        : register_index(register_index),
          to_layer(to_layer),
          id(count++),
          conn(conn),
          second_order_conn(nullptr),
          second_order(false) { }

DendriticNode::~DendriticNode() {
    for (auto& child : children) delete child;
}

void DendriticNode::set_second_order() {
    // Check if leaf node
    if (is_leaf())
        ErrorManager::get_instance()->log_error(
            "Error in dendritic node of " + this->to_layer->str() + "\n"
            "  Leaf dendritic nodes cannot be second order!");

    // This must be set before children are added
    if (children.size() > 1)
        ErrorManager::get_instance()->log_error(
            "Error in dendritic node of " + this->to_layer->str() + "\n"
            "  Dendritic node must be set to second order "
            "before adding children!");

    second_order = true;
}

int DendriticNode::get_second_order_size() const {
    if (not is_second_order())
        ErrorManager::get_instance()->log_error(
            "Error in dendritic node of " + this->to_layer->str() + "\n"
            "  Requested second order size on non-second order node!");
    else if (second_order_conn == nullptr)
        return 0;
    else
        return second_order_conn->get_num_weights();
}

Connection* DendriticNode::get_second_order_connection() const {
    if (not is_second_order())
        ErrorManager::get_instance()->log_error(
            "Error in dendritic node of " + this->to_layer->str() + "\n"
            "  Requested second order host connection on non-second order node!");
    else
        return second_order_conn;
}

DendriticNode* DendriticNode::add_child() {
    if (is_leaf())
        ErrorManager::get_instance()->log_error(
            "Error in dendritic node of " + this->to_layer->str() + "\n"
            "  Dendritic node cannot have children if it has a connection!");
    else if (is_second_order())
        ErrorManager::get_instance()->log_error(
            "Error in dendritic node of " + this->to_layer->str() + "\n"
            "  Second order dendritic nodes cannot have internal children!");

    int child_register = this->register_index;
    if (children.size() > 0) ++child_register;
    auto child = new DendriticNode(child_register, this->to_layer);
    children.push_back(child);
    return child;
}

DendriticNode* DendriticNode::add_child(Connection *conn) {
    if (is_leaf())
        ErrorManager::get_instance()->log_error(
            "Error in dendritic node of " + this->to_layer->str() + "\n"
            "  Dendritic node cannot have children if it has a connection!");

    DendriticNode *child;

    // If adding a connection to a second order node...
    //   If this is the first connection, make it the second order host.
    //   Otherwise, ensure the weights check, and add new node to children.
    if (is_second_order()) {
        if (second_order_conn == nullptr) {
            second_order_conn = conn;
            return this;
        } else if (conn->get_num_weights() != second_order_conn->get_num_weights()) {
            ErrorManager::get_instance()->log_error(
                "Error in dendritic node of " + this->to_layer->str() + "\n"
                "  Second order connections must have identical sizes!");
        } else {
            child = new DendriticNode(this->register_index, this->to_layer, conn);
            children.push_back(child);
        }
    } else {
        child = new DendriticNode(this->register_index, this->to_layer, conn);
        children.push_back(child);
    }
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
