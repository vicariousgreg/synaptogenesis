#include "model/dendritic_node.h"

DendriticNode* DendriticNode::add_child() {
    if (this->conn != NULL)
        ErrorManager::get_instance()->log_error(
            "Dendritic node cannot have children if it has a connection!");
    int child_register = this->register_index;
    if (children.size() > 0) ++child_register;
    auto child = new DendriticNode(child_register, this->to_layer);
    children.push_back(child);
    return child;
}

DendriticNode* DendriticNode::add_child(Connection *conn) {
    if (this->conn != NULL)
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
