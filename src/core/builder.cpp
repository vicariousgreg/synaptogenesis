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
static NoiseConfig *parse_noise_config(Object nco);
static WeightConfig *parse_weight_config(Object wo);
static ArborizedConfig *parse_arborized_config(Object wo);
static SubsetConfig *parse_subset_config(Object wo);

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
 *     -> parse_noise_config
 *     -> parse_dendritic_node
 */
static void parse_layer(Structure *structure, Object lo) {
    std::string name = std::to_string(structure->get_layers().size());
    std::string neural_model = "izhikevich";
    int rows = 0;
    int columns = 0;
    NoiseConfig *noise_config = nullptr;
    bool plastic = true;
    bool global = false;

    std::map<std::string, std::string> properties;

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
        else if (pair.first == "noise")
            noise_config = parse_noise_config(pair.second->get<Object>());
        else if (pair.first != "dendrites") // Skip these till end
            properties[pair.first] = pair.second->get<String>();
    }

    auto layer_config =
        new LayerConfig(name, neural_model, rows, columns, noise_config);

    for (auto pair : properties)
        layer_config->set_property(pair.first, pair.second);

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
 *     -> parse_weight_config
 *     -> parse_arborized_config
 *     -> parse_subset_config
 */
static void parse_connection(Network *network, std::string structure_name, Object co) {
    std::string from_layer = "";
    std::string to_layer = "";
    std::string type_string = "fully connected";
    std::string opcode_string = "add";
    float max_weight = 0.0;
    bool plastic = true;
    int delay = 0;
    std::string dendrite = "root";

    ArborizedConfig *arborized_config = nullptr;
    WeightConfig *weight_config = nullptr;
    SubsetConfig *subset_config = nullptr;

    std::string from_structure = structure_name;
    std::string to_structure = structure_name;

    std::map<std::string, std::string> properties;

    for (auto pair : co.kv_map()) {
        if (pair.first == "from layer")
            from_layer = pair.second->get<String>();
        else if (pair.first == "to layer")
            to_layer = pair.second->get<String>();
        else if (pair.first == "from structure")
            from_structure = pair.second->get<String>();
        else if (pair.first == "to structure")
            to_structure = pair.second->get<String>();
        else if (pair.first == "type")
            type_string = pair.second->get<String>();
        else if (pair.first == "opcode")
            opcode_string = pair.second->get<String>();
        else if (pair.first == "max weight")
            max_weight = std::stof(pair.second->get<String>());
        else if (pair.first == "plastic")
            plastic = pair.second->get<String>() == "true";
        else if (pair.first == "delay")
            delay = std::stoi(pair.second->get<String>());
        else if (pair.first == "weight config")
            weight_config = parse_weight_config(pair.second->get<Object>());
        else if (pair.first == "arborized config")
            arborized_config = parse_arborized_config(pair.second->get<Object>());
        else if (pair.first == "subset config")
            subset_config = parse_subset_config(pair.second->get<Object>());
        else if (pair.first == "dendrite")
            dendrite = pair.second->get<String>();
        else
            properties[pair.first] = pair.second->get<String>();
    }

    ConnectionType type;
    try {
        type = ConnectionTypes[type_string];
    } catch (std::out_of_range) {
        ErrorManager::get_instance()->log_error(
            "Unrecognized connection type: " + type_string);
    }

    Opcode opcode;
    try {
        opcode = Opcodes[opcode_string];
    } catch (std::out_of_range) {
        ErrorManager::get_instance()->log_error(
            "Unrecognized opcode: " + opcode_string);
    }

    auto connection_config =
        new ConnectionConfig(plastic, delay, max_weight,
            type, opcode, weight_config);

    if (arborized_config != nullptr)
        connection_config->set_arborized_config(arborized_config);
    if (subset_config != nullptr)
        connection_config->set_subset_config(subset_config);

    for (auto pair : properties)
        connection_config->set_property(pair.first, pair.second);

    Structure::connect(
        network->get_structure(from_structure),
        from_layer,
        network->get_structure(to_structure),
        to_layer,
        connection_config,
        dendrite);
}

/* Parses a noise configuration */
static NoiseConfig *parse_noise_config(Object nco) {
    if (not has_string(nco, "type"))
        ErrorManager::get_instance()->log_error(
            "No noise config type specified!");

    // Get type string, or use normal as default
    NoiseConfig *noise_config = new NoiseConfig(get_string(nco, "type"));

    // Get properties
    for (auto pair : nco.kv_map())
        if (pair.first != "type")
            noise_config->set_property(pair.first, pair.second->get<String>());

    return noise_config;
}

/* Parses a weight configuration */
static WeightConfig *parse_weight_config(Object wo) {
    if (not has_string(wo, "type"))
        ErrorManager::get_instance()->log_error(
            "No weight config type specified!");

    // Get type string, or use normal as default

    WeightConfig *weight_config = new WeightConfig(get_string(wo, "type"));

    if (has_string(wo, "weight"))
        weight_config->set_property("weight", get_string(wo, "weight"));
    if (has_string(wo, "max weight"))
        weight_config->set_property("max weight", get_string(wo, "max weight"));
    if (has_string(wo, "fraction"))
        weight_config->set_property("fraction", get_string(wo, "fraction"));
    if (has_string(wo, "mean"))
        weight_config->set_property("mean", get_string(wo, "mean"));
    if (has_string(wo, "std dev"))
        weight_config->set_property("std dev", get_string(wo, "std dev"));
    if (has_string(wo, "rows"))
        weight_config->set_property("rows", get_string(wo, "rows"));
    if (has_string(wo, "columns"))
        weight_config->set_property("columns", get_string(wo, "columns"));
    if (has_string(wo, "size"))
        weight_config->set_property("size", get_string(wo, "size"));
    if (has_string(wo, "weight string"))
        weight_config->set_property("weight string",
            get_string(wo, "weight string"));

    if (wo.has<Object>("child"))
        weight_config->set_child(parse_weight_config(wo.get<Object>("child")));

    return weight_config;
}

/* Parses an arborized connection configuration */
static ArborizedConfig *parse_arborized_config(Object wo) {
    int row_field_size = -1;
    int column_field_size = -1;
    int row_stride = 1;
    int column_stride = 1;
    int row_offset = -1;
    int column_offset = -1;
    bool wrap = false;

    for (auto pair : wo.kv_map()) {
        if (pair.first == "row field size")
            row_field_size = std::stoi(pair.second->get<String>());
        else if (pair.first == "column field size")
            column_field_size = std::stoi(pair.second->get<String>());
        else if (pair.first == "field size")
            row_field_size = column_field_size =
                std::stoi(pair.second->get<String>());
        else if (pair.first == "row stride")
            row_stride = std::stoi(pair.second->get<String>());
        else if (pair.first == "column stride")
            column_stride = std::stoi(pair.second->get<String>());
        else if (pair.first == "stride")
            row_stride = column_stride = std::stoi(pair.second->get<String>());
        else if (pair.first == "row offset")
            row_offset = std::stoi(pair.second->get<String>());
        else if (pair.first == "column offset")
            column_offset = std::stoi(pair.second->get<String>());
        else if (pair.first == "offset")
            row_offset = column_offset = std::stoi(pair.second->get<String>());
        else if (pair.first == "wrap")
            wrap = pair.second->get<String>() == "true";
        else
            ErrorManager::get_instance()->log_error(
                "Unrecognized arborized config property: " + pair.first);
    }

    if (row_field_size < 0 or column_field_size < 0)
        ErrorManager::get_instance()->log_error(
            "Unspecified field size for arborized config!");

    if (row_offset < 0 and column_offset < 0) {
        return new ArborizedConfig(
            row_field_size, column_field_size,
            row_stride, column_stride,
            wrap);
    } else if (row_offset < 0 or column_offset < 0) {
        row_offset = std::max(0, row_offset);
        column_offset = std::max(0, column_offset);
    }

    return new ArborizedConfig(
        row_field_size, column_field_size,
        row_stride, column_stride,
        row_offset, column_offset,
        wrap);
}

/* Parses a subset connection configuration */
static SubsetConfig *parse_subset_config(Object wo) {
    int from_row_start = 0;
    int from_row_end = 0;
    int from_col_start = 0;
    int from_col_end = 0;
    int to_row_start = 0;
    int to_row_end = 0;
    int to_col_start = 0;
    int to_col_end = 0;

    for (auto pair : wo.kv_map()) {
        if (pair.first == "from row start")
            from_row_start = std::stoi(pair.second->get<String>());
        else if (pair.first == "from row end")
            from_row_end = std::stoi(pair.second->get<String>());
        else if (pair.first == "from column start")
            from_col_start = std::stoi(pair.second->get<String>());
        else if (pair.first == "from column end")
            from_col_end = std::stoi(pair.second->get<String>());
        else if (pair.first == "to row start")
            to_row_start = std::stoi(pair.second->get<String>());
        else if (pair.first == "to row end")
            to_row_end = std::stoi(pair.second->get<String>());
        else if (pair.first == "to column start")
            to_col_start = std::stoi(pair.second->get<String>());
        else if (pair.first == "to column end")
            to_col_end = std::stoi(pair.second->get<String>());
        else
            ErrorManager::get_instance()->log_error(
                "Unrecognized subset config property: " + pair.first);
    }

    return new SubsetConfig(
        from_row_start, from_row_end,
        from_col_start, from_col_end,
        to_row_start, to_row_end,
        to_col_start, to_col_end);
}

/******************************** WRITING *************************************/
static Object write_structure(Structure *structure);
static Object write_layer(Layer *layer);
static Object write_dendrite(DendriticNode *node);
static Object write_connection(Connection *connection);
static Object write_properties(PropertyConfig *config);
static Object write_weight_config(WeightConfig *weight_config);
static Object write_arborized_config(ArborizedConfig *arborized_config);
static Object write_subset_config(SubsetConfig *subset_config);



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

    for (auto pair : layer->get_config()->get_properties())
        o << pair.first << pair.second;

    auto noise_config = layer->get_config()->noise_config;
    if (noise_config != nullptr)
        o << "noise" << write_properties(noise_config);

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
 *     -> write_weight_config
 *     -> write_arborized_config
 *     -> write_subset_config
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
    o << "weight config"
        << write_weight_config(connection_config->weight_config);

    auto arborized_config = connection_config->get_arborized_config();
    if (arborized_config != nullptr)
        o << "arborized config"
            << write_arborized_config(arborized_config);

    auto subset_config = connection_config->get_subset_config();
    if (subset_config != nullptr)
        o << "subset config" << write_subset_config(subset_config);

    for (auto pair : connection_config->get_properties())
        o << pair.first << pair.second;

    o << "dendrite"
        << connection->to_layer->structure->get_parent_node_name(connection);

    return o;
}

/* Writes a PropertyConfig to an object */
static Object write_properties(PropertyConfig *config) {
    Object o;
    for (auto pair : config->get_properties())
        if (pair.second != "")
            o << pair.first << pair.second;
    return o;
}

/* Writes a weight configuration */
static Object write_weight_config(WeightConfig *weight_config) {
    Object o;

    for (auto pair : weight_config->get_properties())
        o << pair.first << pair.second;

    auto child_config = weight_config->get_child();
    if (child_config != nullptr)
        o << "child" << write_weight_config(child_config);

    return o;
}

/* Writes an arborized connection configuration */
static Object write_arborized_config(ArborizedConfig *arborized_config) {
    Object o;
    o << "row field size" << std::to_string(arborized_config->row_field_size);
    o << "column field size" << std::to_string(arborized_config->column_field_size);
    o << "row stride" << std::to_string(arborized_config->row_stride);
    o << "column stride" << std::to_string(arborized_config->column_stride);
    o << "row offset" << std::to_string(arborized_config->row_offset);
    o << "column offset" << std::to_string(arborized_config->column_offset);

    (arborized_config->wrap)
        ? (o << "wrap" << "true") : (o << "wrap" << "false");

    return o;
}

/* Writes a subset connection configuration */
static Object write_subset_config(SubsetConfig *subset_config) {
    Object o;
    o << "from row start" << subset_config->from_row_start;
    o << "from row end" << subset_config->from_row_end;
    o << "from column start" << subset_config->from_col_start;
    o << "from column end" << subset_config->from_col_end;
    o << "to row start" << subset_config->to_row_start;
    o << "to row end" << subset_config->to_row_end;
    o << "to column start" << subset_config->to_col_start;
    o << "to column end" << subset_config->to_col_end;
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

    auto module_config = new ModuleConfig(
        get_string(mo, "structure"),
        get_string(mo, "layer"),
        get_string(mo, "type"));

    // Get properties
    for (auto pair : mo.kv_map())
        module_config->set_property(pair.first, pair.second->get<String>());

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
