#include <iostream>
#include <fstream>
#include <sstream>

#include "jsonxx/jsonxx.h"

#include "builder.h"
#include "network/network.h"
#include "network/layer_config.h"
#include "network/connection_config.h"
#include "io/environment.h"

using namespace jsonxx;

static PropertyConfig *parse_properties(Object nco) {
    PropertyConfig *config = new PropertyConfig();

    for (auto pair : nco.kv_map()) {
        if (pair.second->is<String>()) {
            config->set(pair.first, pair.second->get<String>());
        } else if (pair.second->is<Object>()) {
            auto props =  parse_properties(pair.second->get<Object>());
            config->set_child(pair.first, props);
            delete props;
        } else if (pair.second->is<Array>()) {
            for (auto item : pair.second->get<Array>().values()) {
                if (item->is<Object>()) {
                    auto props = parse_properties(item->get<Object>());
                    config->add_to_child_array(pair.first, props);
                    delete props;
                } else if (item->is<String>()) {
                    config->add_to_array(pair.first, item->get<String>());
                }
            }
        }
    }

    return config;
}

static Object write_properties(const PropertyConfig *config) {
    Object o;

    // Strings
    for (auto pair : config->get())
        if (pair.second != "")
            o << pair.first << pair.second;

    // Children
    for (auto pair : config->get_children())
        o << pair.first << write_properties(pair.second);

    // Arrays
    for (auto pair : config->get_arrays()) {
        Array a;
        for (auto item : pair.second) a << item;
        if (a.size() > 0) o << pair.first << a;
    }

    // Child Arrays
    for (auto pair : config->get_child_arrays()) {
        Array a;
        for (auto item : pair.second)
            a << write_properties(item);
        if (a.size() > 0) o << pair.first << a;
    }
    return o;
}

Network* load_network(std::string path) {
    std::stringstream buffer;
    buffer << std::ifstream(path).rdbuf();

    Object o;
    o.parse(buffer.str());
    auto props = parse_properties(o);
    Network *network = new Network(new NetworkConfig(props));
    delete props;

    return network;
}

void save_network(Network *network, std::string path) {
    std::ofstream file(path);
	file << write_properties(network->get_config()).json() << std::endl;
	file.close();
}

Environment* load_environment(std::string path) {
    std::stringstream buffer;
    buffer << std::ifstream(path).rdbuf();

    Object o;
    o.parse(buffer.str());

    auto props = parse_properties(o);
    Environment *environment = new Environment(props);
    delete props;

    return environment;
}

void save_environment(Environment *environment, std::string path) {
    std::ofstream file(path);
	file << write_properties(environment).json() << std::endl;
	file.close();
}
