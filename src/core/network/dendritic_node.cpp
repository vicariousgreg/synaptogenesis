#include "network/dendritic_node.h"
#include "network/layer.h"
#include "network/structure.h"

/* Constructor for a root node */
DendriticNode::DendriticNode(Layer *to_layer)
        : parent(nullptr),
          to_layer(to_layer),
          id(std::hash<std::string>()(
              to_layer->structure->name + "/" + to_layer->name + "-0")),
          register_index(0),
          conn(nullptr),
          opcode(ADD),
          second_order_conn(nullptr),
          second_order(false),
          init_val(0.0),
          name("root") { }

/* Constructor for an internal node */
DendriticNode::DendriticNode(DendriticNode *parent, Layer *to_layer,
    int register_index, std::string name, Opcode opcode,
    bool second_order, float init_val)
        : parent(parent),
          to_layer(to_layer),
          id(std::hash<std::string>()(
              to_layer->structure->name + "/" + to_layer->name + "-"
              + std::to_string(to_layer->get_dendritic_nodes().size()))),
          register_index(register_index),
          conn(nullptr),
          opcode(opcode),
          second_order_conn(nullptr),
          second_order(second_order),
          init_val(init_val),
          name(name) { }

/* Constructor for a leaf node */
DendriticNode::DendriticNode(DendriticNode *parent, Layer *to_layer,
    int register_index, Connection *conn)
        : parent(parent),
          to_layer(to_layer),
          id(std::hash<std::string>()(
              to_layer->structure->name + "/" + to_layer->name + "-"
              + std::to_string(to_layer->get_dendritic_nodes().size()))),
          register_index(register_index),
          conn(conn),
          opcode(ADD),
          second_order_conn(nullptr),
          second_order(false),
          init_val(0.0),
          name("Leaf dendrite: " + conn->str()) { }

DendriticNode::~DendriticNode() {
    for (auto& child : children) delete child;
}

int DendriticNode::get_second_order_size() const {
    if (not second_order)
        LOG_ERROR(
            "Error in dendritic node of " + this->to_layer->str() + "\n"
            "  Requested second order size on non-second order node!");
    else if (second_order_conn == nullptr)
        return 0;
    else
        return second_order_conn->get_num_weights();
}

Connection* DendriticNode::get_second_order_connection() const {
    if (not second_order)
        LOG_ERROR(
            "Error in dendritic node of " + this->to_layer->str() + "\n"
            "  Requested second order host connection on non-second order node!");
    else
        return second_order_conn;
}

DendriticNode* DendriticNode::add_child(std::string name, Opcode opcode,
        bool second_order, float init_val) {
    // Verify that this node is not a leaf or a second order node
    if (is_leaf())
        LOG_ERROR(
            "Error in dendritic node of " + this->to_layer->str() + "\n"
            "  Dendritic node cannot have children if it has a connection!");
    else if (this->second_order)
        LOG_ERROR(
            "Error in dendritic node " + name +
            " of " + this->to_layer->str() + "\n"
            "  Second order dendritic nodes cannot have internal children!");

    // Ensure name is not a duplicate
    if (this->to_layer->get_dendritic_node(name, false) != nullptr)
        LOG_ERROR(
            "Error in dendritic node of " + this->to_layer->str() + "\n"
            "  Duplicate dendritic node name: " + name + "!");

    // The child inherits an incremented register index
    auto child =
        new DendriticNode(this, to_layer,
            this->to_layer->get_dendritic_root()->get_max_register_index() + 1,
            name, opcode, second_order, init_val);
    children.push_back(child);
    return child;
}

DendriticNode* DendriticNode::add_child(Connection *conn) {
    // Verify that this node is not a leaf
    if (is_leaf())
        LOG_ERROR(
            "Error in dendritic node " + name +
            " of " + this->to_layer->str() + "\n"
            "  Dendritic node cannot have children if it has a connection!");

    // Verify that the connection targets this node's layer
    if (conn->to_layer != this->to_layer)
        LOG_ERROR(
            "Error in dendritic node " + name +
            " of " + this->to_layer->str() + "\n"
            "  Attempted to add connection to dendritic tree"
            "of the wrong neuron!");

    DendriticNode *child;

    // If adding a connection to a second order node...
    //   If this is the first connection, make it the second order host.
    //   Otherwise, ensure the weights check, and add new node to children.
    if (second_order) {
        if (second_order_conn == nullptr) {
            second_order_conn = conn;
            return this;
        } else if (conn->get_num_weights() != second_order_conn->get_num_weights()) {
            LOG_ERROR(
                "Error in dendritic node of " + this->to_layer->str() + "\n"
                "  Second order connections must have identical sizes!");
        } else {
            child =
                new DendriticNode(this, to_layer, register_index, conn);
            children.push_back(child);
        }
    } else {
        child =
            new DendriticNode(this, to_layer, register_index, conn);
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

std::string DendriticNode::str() const {
    return "[Dendrite: " + name + " (" + to_layer->name + ")]";
}
