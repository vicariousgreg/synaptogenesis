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

/******************************************************************************/
/******************************** NETWORK *************************************/
/******************************************************************************/

/******************************** PARSING *************************************/
static bool has_string(Object o, std::string key) { return o.has<String>(key); }
static std::string get_string(Object o, std::string key, std::string def_val="")
    { return (o.has<String>(key)) ? o.get<String>(key) : def_val; }

static bool has_object(Object o, std::string key) { return o.has<Object>(key); }
static bool has_array(Object o, std::string key) { return o.has<Array>(key); }

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
                    config->add_to_array(pair.first, props);
                    delete props;
                }
            }
        }
    }

    return config;
}

/* Top level network parser */
Network* load_network(std::string path) {
    std::ifstream file("networks/" + path);
    std::stringstream buffer;

    buffer << file.rdbuf();
    std::string str = buffer.str();

    Object o;
    o.parse(str);
    auto props = parse_properties(o);
    Network *network = new Network(new NetworkConfig(props));
    delete props;

    return network;
}

/******************************** WRITING *************************************/
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
        for (auto item : pair.second)
            a << write_properties(item);
        if (a.size() > 0) o << pair.first << a;
    }
    return o;
}

void save_network(Network *network, std::string path) {
    std::ofstream file("networks/" + path);

	Object o;
	Array struct_array;
	Array conn_array;
	for (auto structure : network->get_structures()) {
	    struct_array << write_properties(structure->config);
        if (structure->get_connections().size() > 0)
            for (auto connection : structure->get_connections())
                conn_array << write_properties(connection->config);
    }
	o << "structures" << struct_array;
	o << "connections" << conn_array;

	file << o.json() << std::endl;
	file.close();
}

/******************************************************************************/
/******************************* ENVIRONMENT ***********************************/
/******************************************************************************/

/******************************** PARSING *************************************/
static ModuleConfig* parse_module(Object mo);


Environment* load_environment(std::string path) {
    std::ifstream file("environments/" + path);
    std::stringstream buffer;

    buffer << file.rdbuf();
    std::string str = buffer.str();

    Environment *environment = new Environment();

    Object o;
    o.parse(str);
    if (has_array(o, "modules")) {
        for (auto module : o.get<Array>("modules").values()) {
            auto config = parse_module(module->get<Object>());
            if (config != nullptr)
                environment->add_module(config);
        }
    }

    return environment;
}

static ModuleConfig* parse_module(Object mo) {
    if (get_string(mo, "skip", "false") == "true") return nullptr;

    if (not has_string(mo, "type"))
        ErrorManager::get_instance()->log_error(
            "No module type specified!");

    if (not has_array(mo, "layers"))
        ErrorManager::get_instance()->log_error(
            "No layers for module specified!");

    auto props = parse_properties(mo);
    auto module_config = new ModuleConfig(props);
    delete props;

    return module_config;
}

/******************************** WRITING *************************************/
void save_environment(Environment *environment, std::string path) {
    std::ofstream file("environments/" + path);

	Object o;
	Array a;
	for (auto module : environment->get_modules())
	    a << write_properties(module);
	o << "modules" << a;

	file << o.json() << std::endl;
	file.close();
}
