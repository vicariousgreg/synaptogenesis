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
static void parse_structure(Network *network, Object so);
static void parse_connection(Network *network, std::string structure_name, Object co);
static PropertyConfig *parse_properties(Object nco);

static bool has_string(Object o, std::string key) { return o.has<String>(key); }
static std::string get_string(Object o, std::string key, std::string def_val="")
    { return (o.has<String>(key)) ? o.get<String>(key) : def_val; }

static bool has_object(Object o, std::string key) { return o.has<Object>(key); }
static bool has_array(Object o, std::string key) { return o.has<Array>(key); }


/* Top level network parser
 *     -> parse_structure
 *     -> parse_connection
 */
Network* load_network(std::string path) {
    std::ifstream file("networks/" + path);
    std::stringstream buffer;

    buffer << file.rdbuf();
    std::string str = buffer.str();

    Network *network = new Network();

    Object o;
    o.parse(str);
    if (has_array(o, "structures")) {
        // Parse structures
        for (auto structure : o.get<Array>("structures").values())
            parse_structure(network, structure->get<Object>());

        // Parse connections after because of inter-structure connections
        for (auto structure : o.get<Array>("structures").values()) {
            auto so = structure->get<Object>();
            if (has_array(so, "connections"))
                for (auto connection : so.get<Array>("connections").values())
                    parse_connection(network, so.get<String>("name"),
                        connection->get<Object>());
        }
    }

    return network;
}

/* Parses a structure
 *     -> parse_properties
 */
static void parse_structure(Network *network, Object so) {
    std::string name = get_string(so, "name",
        std::to_string(network->get_structures().size()));

    Structure *structure;

    // Get cluster type string if it exists
    if (has_string(so, "cluster type")) {
        std::string cluster_type_string
            = get_string(so, "cluster type", "parallel");

        // Convert string to ClusterType
        ClusterType cluster_type;
        try {
            cluster_type = ClusterTypes[cluster_type_string];
        } catch (std::out_of_range) {
            ErrorManager::get_instance()->log_error(
                "Unrecognized cluster type for structure " + name
                + ": " + cluster_type_string);
        }
        structure = new Structure(name, cluster_type);
    } else {
        structure = new Structure(name);
    }

    if (has_array(so, "layers")) {
        for (auto layer : so.get<Array>("layers").values()) {
            auto props = parse_properties(layer->get<Object>());
            structure->add_layer(new LayerConfig(props));
            delete props;
        }
    }

    network->add_structure(structure);
}

/* Parses a connection
 *     -> parse_properties
 */
static void parse_connection(Network *network, std::string structure_name, Object co) {
    PropertyConfig* props = parse_properties(co);

    try {
        ConnectionTypes[props->get("type", "fully connected")];
    } catch (std::out_of_range) {
        ErrorManager::get_instance()->log_error(
            "Unrecognized connection type: " + props->get("type"));
    }

    try {
        Opcodes[props->get("opcode", "add")];
    } catch (std::out_of_range) {
        ErrorManager::get_instance()->log_error(
            "Unrecognized opcode: " + props->get("opcode"));
    }

    auto connection_config = new ConnectionConfig(props);

    Structure::connect(
        network->get_structure(props->get("from structure", structure_name)),
        props->get("from layer", ""),
        network->get_structure(props->get("to structure", structure_name)),
        props->get("to layer", ""),
        connection_config,
        props->get("dendrite", "root"),
        props->get("name", ""));

    delete props;
}

static PropertyConfig *parse_properties(Object nco) {
    PropertyConfig *config = new PropertyConfig();

    for (auto pair : nco.kv_map()) {
        if (pair.second->is<String>()) {
            config->set_value(pair.first, pair.second->get<String>());
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

/******************************** WRITING *************************************/
static Object write_structure(Structure *structure);
static Object write_layer(Layer *layer);
static void write_dendrites(Object& parent, std::string key,
        const DendriticNodeList& children);
static Object write_connection(Connection *connection);
static Object write_properties(PropertyConfig *config);


/* Top level network writer
 *     -> write_structure
 */
void save_network(Network *network, std::string path) {
    std::ofstream file("networks/" + path);

	Object o;
	Array a;
	for (auto structure : network->get_structures())
	    a << write_structure(structure);
	o << "structures" << a;

	file << o.json() << std::endl;
	file.close();
}

/* Writes a structure
 *     -> write_layer
 *     -> write_connection
 */
static Object write_structure(Structure *structure) {
    Object o;
    o << "name" << structure->name;
    o << "cluster type" << ClusterTypeStrings[structure->cluster_type];

    if (structure->get_layers().size() > 0) {
        Array a;
        for (auto layer : structure->get_layers())
            a << write_layer(layer);
        o << "layers" << a;
    }

    if (structure->get_connections().size() > 0) {
        Array a;
        for (auto connection : structure->get_connections())
            a << write_connection(connection);
        o << "connections" << a;
    }

    return o;
}

/* Writes a layer
 *     -> write_noise_config
 *     -> write_dendrites
 */
static Object write_layer(Layer *layer) {
    Object o;
    o << "name" << layer->name;
    o << "neural model" << layer->neural_model;
    o << "rows" << std::to_string(layer->rows);
    o << "columns" << std::to_string(layer->columns);

    (layer->plastic)
        ? (o << "plastic" << "true") : (o << "plastic" << "false");
    (layer->global)
        ? (o << "global" << "true") : (o << "global" << "false");

    for (auto pair : layer->get_config()->get())
        o << pair.first << pair.second;

    auto noise_config = layer->get_config()->get_child("noise config", nullptr);
    if (noise_config != nullptr)
        o << "noise config" << write_properties(noise_config);

    write_dendrites(o, "dendrites", layer->dendritic_root->get_children());

    return o;
}

/* Writes a dendrite */
static void write_dendrites(Object& parent, std::string key,
        const DendriticNodeList& children) {
    Array a;
    for (auto child : children) {
        if (not child->is_leaf()) {
            Object o;
            o << "name" << child->name;
            if (child->is_second_order())
                o << "second order" << "true";
            else
                o << "second order" << "false";
            write_dendrites(o, "children", child->get_children());
            a << o;
        }
    }

    if (a.size() > 0) parent << key << a;
}

/* Writes a connection
 *     -> write_properties
 */
static Object write_connection(Connection *connection) {
    Object o;

    o << "from layer" << connection->from_layer->name;
    o << "to layer" << connection->to_layer->name;
    o << "type" << ConnectionTypeStrings[connection->type];
    o << "opcode" << OpcodeStrings[connection->opcode];
    o << "max weight" << std::to_string(connection->max_weight);
    o << "delay" << std::to_string(connection->delay);

    if (connection->from_layer->structure != connection->to_layer->structure) {
        o << "from structure" << connection->from_layer->structure->name;
        o << "to structure" << connection->to_layer->structure->name;
    }

    (connection->plastic)
        ? (o << "plastic" << "true") : (o << "plastic" << "false");

    auto connection_config = connection->get_config();

    o << "dendrite"
        << connection->to_layer->structure->get_parent_node_name(connection);

    for (auto pair : connection_config->get())
        o << pair.first << pair.second;
    for (auto pair : connection_config->get_children())
        o << pair.first << write_properties(pair.second);
    for (auto pair : connection_config->get_arrays()) {
        Array a;
        for (auto item : pair.second)
            a << write_properties(item);
        o << pair.first << a;
    }

    return o;
}

/* Writes a PropertyConfig to an object */
static Object write_properties(PropertyConfig *config) {
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
            o << pair.first << write_properties(item);
        if (a.size() > 0) o << pair.first << a;
    }
    return o;
}

/******************************************************************************/
/******************************* ENVIRONMENT ***********************************/
/******************************************************************************/

/******************************** PARSING *************************************/
static ModuleConfig* parse_module(Object mo);


/* Top level environment parser
 *     -> parse_module
 */
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

/* Parses a module */
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
/* Top level environment writer
 *     -> write_properties (defined above)
 */
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
