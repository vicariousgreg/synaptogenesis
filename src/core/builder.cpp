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
static void parse_layer(Structure *structure, Object lo);
static void parse_dendrite(Structure *structure, std::string layer,
    std::string node, Object dobj);
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
    if (o.has<Array>("structures")) {
        // Parse structures
        for (auto structure : o.get<Array>("structures").values())
            parse_structure(network, structure->get<Object>());

        // Parse connections after because of inter-structure connections
        for (auto structure : o.get<Array>("structures").values()) {
            auto so = structure->get<Object>();
            if (so.has<Array>("connections"))
                for (auto connection : so.get<Array>("connections").values())
                    parse_connection(network, so.get<String>("name"),
                        connection->get<Object>());
        }
    }

    return network;
}

/* Parses a structure
 *     -> parse_layer
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

    if (so.has<Array>("layers"))
        for (auto layer : so.get<Array>("layers").values())
            parse_layer(structure, layer->get<Object>());

    network->add_structure(structure);
}

/* Parses a layer
 *     -> parse_properties
 *     -> parse_dendritic_node
 */
static void parse_layer(Structure *structure, Object lo) {
    std::string name = std::to_string(structure->get_layers().size());
    std::string neural_model = "izhikevich";
    int rows = 0;
    int columns = 0;
    PropertyConfig *noise_config = nullptr;
    bool plastic = true;
    bool global = false;

    StringPairList properties;

    for (auto pair : lo.kv_map()) {
        if (pair.first == "name")
            name = pair.second->get<String>();
        else if (pair.first == "neural model")
            neural_model = pair.second->get<String>();
        else if (pair.first == "rows")
            rows = std::stoi(pair.second->get<String>());
        else if (pair.first == "columns")
            columns = std::stoi(pair.second->get<String>());
        else if (pair.first == "plastic")
            plastic = pair.second->get<String>() == "true";
        else if (pair.first == "global")
            global = pair.second->get<String>() == "true";
        else if (pair.first == "noise config")
            noise_config = parse_properties(pair.second->get<Object>());
        else if (pair.first != "dendrites") // Skip these till end
            properties.push_back(StringPair(pair.first, pair.second->get<String>()));
    }

    auto layer_config =
        new LayerConfig(name, neural_model, rows, columns, noise_config);

    for (auto pair : properties)
        layer_config->set(pair.first, pair.second);

    structure->add_layer(layer_config);

    if (has_object(lo, "dendrites"))
        parse_dendrite(structure, name, "root", lo.get<Object>("dendrites"));
}

/* Parses a dendritic node
 *     -> parse_dendritic_node (recursive)
 */
static void parse_dendrite(Structure *structure, std::string layer,
        std::string node, Object dobj) {
    if (get_string(dobj, "second order", "false") == "true")
        structure->set_second_order(layer, node);

    if (has_array(dobj, "children")) {
        for (auto child : dobj.get<Array>("children").values()) {
            auto child_obj = child->get<Object>();

            if (not has_string(child_obj, "name"))
                ErrorManager::get_instance()->log_error(
                    "Unspecifed name for dendritic node in layer: " + layer);

            auto child_name = get_string(child_obj, "name");

            structure->create_dendritic_node(layer, node, child_name);
            parse_dendrite(structure, layer, child_name, child_obj);
        }
    }
}

/* Parses a connection
 *     -> parse_properties
 */
static void parse_connection(Network *network, std::string structure_name, Object co) {
    std::string name = "";
    std::string from_layer = "";
    std::string to_layer = "";
    std::string dendrite = "root";

    PropertyConfig *arborized_config = nullptr;
    PropertyConfig *weight_config = nullptr;
    PropertyConfig *subset_config = nullptr;

    std::string from_structure = structure_name;
    std::string to_structure = structure_name;

    PropertyConfig* props = new PropertyConfig();

    for (auto pair : co.kv_map()) {
        if (pair.first == "name")
            name = pair.second->get<String>();
        else if (pair.first == "from layer")
            from_layer = pair.second->get<String>();
        else if (pair.first == "to layer")
            to_layer = pair.second->get<String>();
        else if (pair.first == "from structure")
            from_structure = pair.second->get<String>();
        else if (pair.first == "to structure")
            to_structure = pair.second->get<String>();
        else if (pair.first == "dendrite")
            dendrite = pair.second->get<String>();
        else if (pair.second->is<Object>())
            props->set_child(pair.first,
                parse_properties(pair.second->get<Object>()));
        else if (pair.second->is<String>())
            props->set_value(pair.first, pair.second->get<String>());
    }

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
    delete props;

    Structure::connect(
        network->get_structure(from_structure),
        from_layer,
        network->get_structure(to_structure),
        to_layer,
        connection_config,
        dendrite,
        name);
}

static PropertyConfig *parse_properties(Object nco) {
    PropertyConfig *config = new PropertyConfig();

    for (auto pair : nco.kv_map())
        config->set_value(pair.first, pair.second->get<String>());

    return config;
}

/******************************** WRITING *************************************/
static Object write_structure(Structure *structure);
static Object write_layer(Layer *layer);
static Object write_dendrite(DendriticNode *node);
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
 *     -> write_dendrite
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

    o << "dendrites" << write_dendrite(layer->dendritic_root);

    return o;
}

/* Writes a dendrite */
static Object write_dendrite(DendriticNode *node) {
    Object o;
    o << "name" << node->name;
    if (node->is_second_order())
        o << "second order" << "true";
    else
        o << "second order" << "false";

    Array a;
    for (auto child : node->get_children())
        if (not child->is_leaf())
            a << write_dendrite(child);
    if (a.size() > 0) o << "children" << a;

    return o;
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

    return o;
}

/* Writes a PropertyConfig to an object */
static Object write_properties(PropertyConfig *config) {
    Object o;
    for (auto pair : config->get())
        if (pair.second != "")
            o << pair.first << pair.second;
    for (auto pair : config->get_children())
        o << pair.first << write_properties(pair.second);
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
    if (o.has<Array>("modules")) {
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

    auto module_config = new ModuleConfig(get_string(mo, "type"));

    if (not has_array(mo, "layers"))
        ErrorManager::get_instance()->log_error(
            "No layers for module specified!");

    // Parse layers
    for (auto layer : mo.get<Array>("layers").values()) {
        StringPairList layer_props;
        for (auto pair : layer->get<Object>().kv_map())
            layer_props.push_back(
                StringPair(pair.first, pair.second->get<String>()));
        module_config->add_layer(new PropertyConfig(layer_props));
    }

    // Get properties
    for (auto pair : mo.kv_map())
        if (pair.first != "layers")
            module_config->set(
                pair.first, pair.second->get<String>());

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
	for (auto module : environment->get_modules()) {
	    Array l;
	    auto module_o = write_properties(module);
	    for (auto layer : module->get_layers())
	        l << write_properties(layer);
	    module_o << "layers" << l;
	    a << module_o;
    }
	o << "modules" << a;

	file << o.json() << std::endl;
	file.close();
}
