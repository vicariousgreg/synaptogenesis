#include <iostream>
#include <fstream>
#include <sstream>

#include "jsonxx/jsonxx.h"

#include "model/model_builder.h"
#include "model/layer_config.h"
#include "model/connection_config.h"
#include "io/module/module.h"

using namespace jsonxx;

/******************************************************************************/
/******************************** PARSING *************************************/
/******************************************************************************/

static void parse_structure(Model *model, Object so);
static void parse_layer(Structure *structure, Object lo);
static void parse_connection(Structure *structure, Object co);
static ModuleConfig* parse_module(Object mo);
static NoiseConfig *parse_noise_config(Object nco);
static WeightConfig *parse_weight_config(Object wo);
static ArborizedConfig *parse_arborized_config(Object wo);
static SubsetConfig *parse_subset_config(Object wo);

static std::string get_string(Object o, std::string key,
        std::string default_value) {
    if (o.has<String>(key))
        return o.get<String>(key);
    else return default_value;
}



/* Top level model parser
 *     -> parse_structure
 */
Model* load_model(std::string path) {
    std::ifstream file("models/" + path);
    std::stringstream buffer;

    buffer << file.rdbuf();
    std::string str = buffer.str();

    Model *model = new Model();

    Object o;
    o.parse(str);
    if (o.has<Array>("structures"))
        for (auto structure : o.get<Array>("structures").values())
            parse_structure(model, structure->get<Object>());

    return model;
}

/* Parses a structure
 *     -> parse_layer
 *     -> parse_connection
 */
static void parse_structure(Model *model, Object so) {
    std::string name = get_string(so, "name",
        std::to_string(model->get_structures().size()));

    // Get cluster type string, or use parallel as default
    std::string cluster_type_string
        = get_string(so, "cluster type", "parallel");

    // Convert string to ClusterType
    ClusterType cluster_type;
    if (cluster_type_string == "parallel")
        cluster_type = PARALLEL;
    else if (cluster_type_string == "sequential")
        cluster_type = SEQUENTIAL;
    else if (cluster_type_string == "feedforward")
        cluster_type = FEEDFORWARD;
    else
        ErrorManager::get_instance()->log_error(
            "Unrecognized cluster type for structure " + name
            + ": " + cluster_type_string);

    Structure *structure = new Structure(name, cluster_type);

    if (so.has<Array>("layers"))
        for (auto layer : so.get<Array>("layers").values())
            parse_layer(structure, layer->get<Object>());
    if (so.has<Array>("connections"))
        for (auto connection : so.get<Array>("connections").values())
            parse_connection(structure, connection->get<Object>());

    model->add_structure(structure);
}

/* Parses a layer
 *     -> parse_noise_config
 *     -> parse_module
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
    std::vector<ModuleConfig*> modules;

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
        else if (pair.first == "modules")
            for (auto val : pair.second->get<Array>().values())
                modules.push_back(parse_module(val->get<Object>()));
        else
            properties[pair.first] = pair.second->get<String>();
    }

    auto layer_config =
        new LayerConfig(name, neural_model, rows, columns, noise_config);

    for (auto pair : properties)
        layer_config->set_property(pair.first, pair.second);

    structure->add_layer(layer_config);
    for (auto module : modules)
        if (module != nullptr)
            structure->add_module(name, module);
}

/* Parses a connection
 *     -> parse_weight_config
 *     -> parse_arborized_config
 *     -> parse_subset_config
 */
static void parse_connection(Structure *structure, Object co) {
    std::string from_layer = "";
    std::string to_layer = "";
    std::string type_string = "fully connected";
    std::string opcode_string = "add";
    float max_weight = 0.0;
    bool plastic = true;
    int delay = 0;

    ArborizedConfig *arborized_config = nullptr;
    WeightConfig *weight_config = nullptr;
    SubsetConfig *subset_config = nullptr;

    for (auto pair : co.kv_map()) {
        if (pair.first == "from layer")
            from_layer = pair.second->get<String>();
        else if (pair.first == "to layer")
            to_layer = pair.second->get<String>();
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
        else
            ErrorManager::get_instance()->log_error(
                "Unrecognized connection property: " + pair.first);
    }

    ConnectionType type;
    if (type_string == "fully connected")    type = FULLY_CONNECTED;
    else if (type_string == "subset")        type = SUBSET;
    else if (type_string == "one to one")    type = ONE_TO_ONE;
    else if (type_string == "convergent")    type = CONVERGENT;
    else if (type_string == "convolutional") type = CONVOLUTIONAL;
    else if (type_string == "divergent")     type = DIVERGENT;
    else
        ErrorManager::get_instance()->log_error(
            "Unrecognized connection type: " + type_string);

    Opcode opcode;
    if (opcode_string == "add")           opcode = ADD;
    else if (opcode_string == "sub")      opcode = SUB;
    else if (opcode_string == "mult")     opcode = MULT;
    else if (opcode_string == "div")      opcode = DIV;
    else if (opcode_string == "pool")     opcode = POOL;
    else if (opcode_string == "reward")   opcode = REWARD;
    else if (opcode_string == "modulate") opcode = MODULATE;
    else
        ErrorManager::get_instance()->log_error(
            "Unrecognized opcode: " + opcode_string);

    auto connection_config =
        new ConnectionConfig(plastic, delay, max_weight,
            type, opcode, weight_config);

    if (arborized_config != nullptr)
        connection_config->set_arborized_config(arborized_config);
    if (subset_config != nullptr)
        connection_config->set_subset_config(subset_config);

    structure->connect_layers(from_layer, to_layer, connection_config);
}

/* Parses a module list */
static ModuleConfig* parse_module(Object mo) {
    // Get type string, or use normal as default
    std::string type = get_string(mo, "type", "");
    if (type == "")
        ErrorManager::get_instance()->log_error(
            "No module type specified!");

    if (get_string(mo, "skip", "false") == "true") return nullptr;

    auto module_config = new ModuleConfig(type);

    // Get properties
    for (auto pair : mo.kv_map())
        if (pair.first != "type")
            module_config->set_property(pair.first, pair.second->get<String>());

    return module_config;
}

/* Parses a noise configuration */
static NoiseConfig *parse_noise_config(Object nco) {
    NoiseConfig *noise_config;

    // Get type string, or use normal as default
    noise_config = new NoiseConfig(get_string(nco, "type", "normal"));

    // Get properties
    for (auto pair : nco.kv_map())
        if (pair.first != "type")
            noise_config->set_property(pair.first, pair.second->get<String>());

    return noise_config;
}

/* Parses a weight configuration */
static WeightConfig *parse_weight_config(Object wo) {
    // Get type string, or use normal as default
    std::string type_string = get_string(wo, "type", "flat");

    auto weight = std::stof(get_string(wo, "weight", "1.0"));
    auto max_weight = std::stof(get_string(wo, "max weight", "1.0"));
    auto fraction = std::stof(get_string(wo, "fraction", "1.0"));
    auto mean = std::stof(get_string(wo, "mean", "1.0"));
    auto std_dev = std::stof(get_string(wo, "std dev", "0.3"));
    auto rows = std::stoi(get_string(wo, "rows", "0"));
    auto columns = std::stoi(get_string(wo, "columns", "0"));
    auto weight_string = get_string(wo, "weight string", "");

    Object child;
    if (wo.has<Object>("child")) child = wo.get<Object>("child");

    WeightConfig *weight_config;

    // Convert string to type
    if (type_string == "flat") {
        weight_config = new FlatWeightConfig(weight, fraction);
    } else if (type_string == "random") {
        weight_config = new RandomWeightConfig(max_weight, fraction);
    } else if (type_string == "gaussian") {
        weight_config = new GaussianWeightConfig(mean, std_dev, fraction);
    } else if (type_string == "log normal") {
        weight_config = new LogNormalWeightConfig(mean, std_dev, fraction);
    } else if (type_string == "surround") {
        weight_config =
            new SurroundWeightConfig(rows, columns, parse_weight_config(child));
    } else if (type_string == "specified") {
        weight_config = new SpecifiedWeightConfig(weight_string);
    } else {
        ErrorManager::get_instance()->log_error(
            "Unrecognized noise type: " + type_string);
    }

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

/******************************************************************************/
/******************************** WRITING *************************************/
/******************************************************************************/

static Object write_structure(Structure *structure);
static Object write_layer(Layer *layer);
static Object write_connection(Connection *connection);
static Object write_module(ModuleConfig *module_config);
static Object write_noise_config(NoiseConfig *noise_config);
static Object write_weight_config(WeightConfig *weight_config);
static Object write_arborized_config(ArborizedConfig *arborized_config);
static Object write_subset_config(SubsetConfig *subset_config);



/* Top level model writer
 *     -> write_structure
 */
void save_model(Model *model, std::string path) {
    std::ofstream file("models/" + path);

	Object o;
	Array structures;
	for (auto structure : model->get_structures())
	    o << "structures" << write_structure(structure);

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

    switch (structure->cluster_type) {
        case (PARALLEL): o << "cluster type" << "parallel"; break;
        case (SEQUENTIAL): o << "cluster type" << "sequential"; break;
        case (FEEDFORWARD): o << "cluster type" << "feedforward"; break;
    }

    if (structure->get_layers().size() > 0) {
        Array a;
        for (auto layer : structure->get_layers()) {
            a << write_layer(layer);
        }
        o << "layers" << a;
    }

    if (structure->get_connections().size() > 0) {
        Array a;
        for (auto connection : structure->get_connections()) {
            a << write_connection(connection);
        }
        o << "connections" << a;
    }

    return o;
}

/* Writes a layer
 *     -> write_noise_config
 *     -> write_module
 */
static Object write_layer(Layer *layer) {
    Object o;
    o << "name" << layer->name;
    o << "neural model" << layer->neural_model;
    o << "rows" << std::to_string(layer->rows);
    o << "columns" << std::to_string(layer->columns);
    if (layer->plastic)
        o << "plastic" << "true";
    else
        o << "plastic" << "false";
    if (layer->global)
        o << "global" << "true";
    else
        o << "global" << "false";

    for (auto pair : layer->get_config()->get_properties())
        o << pair.first << pair.second;

    auto noise_config = layer->get_config()->noise_config;
    if (noise_config != nullptr)
        o << "noise" << write_noise_config(noise_config);

    if (layer->get_module_configs().size() > 0) {
        Array a;
        for (auto module : layer->get_module_configs())
            a << write_module(module);
        o << "modules" << a;
    }

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

    switch (connection->type) {
        case (FULLY_CONNECTED): o << "type" << "fully connected"; break;
        case (SUBSET):          o << "type" << "subset"; break;
        case (ONE_TO_ONE):      o << "type" << "one to one"; break;
        case (CONVERGENT):      o << "type" << "convergent"; break;
        case (CONVOLUTIONAL):   o << "type" << "convolutional"; break;
        case (DIVERGENT):       o << "type" << "divergent"; break;
    }

    switch (connection->opcode) {
        case (ADD):      o << "opcode" << "add"; break;
        case (SUB):      o << "opcode" << "sub"; break;
        case (MULT):     o << "opcode" << "mult"; break;
        case (DIV):      o << "opcode" << "div"; break;
        case (POOL):     o << "opcode" << "pool"; break;
        case (REWARD):   o << "opcode" << "reward"; break;
        case (MODULATE): o << "opcode" << "modulate"; break;
    }

    o << "max weight" << std::to_string(connection->max_weight);
    if (connection->plastic)
        o << "plastic" << "true";
    else
        o << "plastic" << "false";
    o << "delay" << std::to_string(connection->delay);

    auto connection_config = connection->get_config();
    o << "weight config" << write_weight_config(connection_config->weight_config);

    auto arborized_config = connection_config->get_arborized_config();
    if (arborized_config != nullptr)
        o << "arborized config"
            << write_arborized_config(arborized_config);

    auto subset_config = connection_config->get_subset_config();
    if (subset_config != nullptr)
        o << "subset config"
            << write_subset_config(subset_config);

    return o;
}

/* Writes a module list */
static Object write_module(ModuleConfig *module_config) {
    Object o;
    for (auto pair : module_config->get_properties())
        o << pair.first << pair.second;

    return o;
}

/* Writes a noise configuration */
static Object write_noise_config(NoiseConfig *noise_config) {
    Object o;

    for (auto pair : noise_config->get_properties())
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

    o << "row field size" << arborized_config->row_field_size;
    o << "column field size" << arborized_config->column_field_size;
    o << "row stride" << arborized_config->row_stride;
    o << "column stride" << arborized_config->column_stride;
    o << "row offset" << arborized_config->row_offset;
    o << "column offset" << arborized_config->column_offset;
    o << "wrap" << arborized_config->wrap;

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
