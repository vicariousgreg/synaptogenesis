#include "model/dendritic_node.h"

void DendriticNode::add_child() {
    if (this->conn != NULL)
        ErrorManager::get_instance()->log_error(
            "Dendritic node cannot have children if it has a connection!");
    int child_register = this->register_index;
    if (children.size() > 0) ++child_register;
    children.push_back(DendriticNode(child_register, this->to_layer));
}

void DendriticNode::add_child(Connection *conn) {
    if (this->conn != NULL)
        ErrorManager::get_instance()->log_error(
            "Dendritic node cannot have children if it has a connection!");
    children.push_back(DendriticNode(this->register_index, this->to_layer, conn));
}
