#include <queue>

#include "network/layer.h"
#include "network/structure.h"
#include "network/connection.h"
#include "util/error_manager.h"

Layer::Layer(Structure *structure, const LayerConfig *config)
        : config(config),
          name(config->name),
          id(std::hash<std::string>()(structure->name + "/" + name)),
          neural_model(config->neural_model),
          structure(structure),
          rows(config->rows),
          columns(config->columns),
          size(rows * columns),
          plastic(config->plastic),
          global(config->global),
          dendritic_root(new DendriticNode(this)) {
    add_dendrites("root", config->get_array("dendrites"));
}

Layer::~Layer() {
    delete dendritic_root;
    delete config;
}

const ConnectionList& Layer::get_input_connections() const
    { return input_connections; }
const ConnectionList& Layer::get_output_connections() const
    { return output_connections; }

DendriticNodeList Layer::get_dendritic_nodes() const {
    DendriticNodeList nodes;
    std::queue<DendriticNode*> q;
    q.push(this->dendritic_root);

    while (not q.empty()) {
        auto curr = q.front();
        q.pop();
        nodes.push_back(curr);
        for (auto child : curr->get_children())
            q.push(child);
    }

    return nodes;
}

DendriticNode* Layer::get_dendritic_node(std::string name,
        bool log_error) const {
    for (auto node: get_dendritic_nodes())
        if (node->name == name) return node;

    if (log_error)
        ErrorManager::get_instance()->log_error(
            "Error in " + this->str() + ":\n"
            "  Cannot find Dendritic Node " + name + "!");

    return nullptr;
}

std::string Layer::get_parameter(std::string key, std::string def_val) const {
    if (not this->get_config()->has(key))
        ErrorManager::get_instance()->log_warning(
            "Error in " + this->str() + ":\n"
            "  Unspecified parameter: " + key
            + " -- using " + def_val + ".");
    return this->get_config()->get(key, def_val);
}

int Layer::get_max_delay() const {
    int max_delay = 0;

    // Determine max delay for output connections
    for (auto& conn : get_output_connections()) {
        int delay = conn->delay;
        if (delay > max_delay)
            max_delay = delay;
    }
    return max_delay;
}

int Layer::get_num_weights() const {
    int num_weights = 0;
    for (auto& conn : get_input_connections())
        num_weights += conn->get_num_weights();
    return num_weights;
}

void Layer::add_input_connection(Connection* connection) {
    this->input_connections.push_back(connection);
}

void Layer::add_output_connection(Connection* connection) {
    this->output_connections.push_back(connection);
}

std::string Layer::str() const {
    return "[Layer: " + name + " (" + structure->name + ")]";
}

void Layer::add_dendrites(std::string parent_name,
        const ConfigArray& dendrites) {
    auto parent = get_dendritic_node(parent_name);
    for (auto dendrite : dendrites) {
        if (not dendrite->has("name"))
            ErrorManager::get_instance()->log_error(
                "Attempted to dendrite without name to layer!");

        bool second_order = dendrite->get_bool("second order", false);
        auto child = parent->add_child(dendrite->get("name"), second_order);

        add_dendrites(child->name,
            dendrite->get_array("children"));
    }
}

int get_num_connections(const LayerList& layers) {
    int num_connections = 0;
    for (auto& layer : layers)
        num_connections += layer->get_input_connections().size();
    return num_connections;
}

bool check_equal_sizes(const LayerList& layers) {
    if (layers.size() == 0) return true;
    int size = layers.at(0)->size;
    for (auto& layer : layers)
        if (layer->size != size) return false;
    return true;
}
