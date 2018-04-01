#include <bitset>

#include "io/module.h"
#include "network/network.h"
#include "network/structure.h"
#include "network/layer.h"

ModuleConfig::ModuleConfig(PropertyConfig *config)
    : PropertyConfig(config) { }

ModuleConfig::ModuleConfig(std::string type) {
    this->set("type", type);
}

ModuleConfig::ModuleConfig(std::string type,
        std::string structure, std::string layer) {
    this->set("type", type);
    this->add_layer(structure, layer);
}

ModuleConfig* ModuleConfig::add_layer(std::string structure,
        std::string layer) {
    auto props = new PropertyConfig(
        { {"structure", structure},
          {"layer", layer} });
    add_layer(props);
    delete props;
    return this;
}

ModuleConfig* ModuleConfig::add_layer(PropertyConfig *config) {
    if (not config->has("structure") or not config->has("layer"))
        LOG_ERROR(
            "Module layer config must have structure and layer name!");
    this->add_to_child_array("layers", config);
    return this;
}

const PropertyConfig* ModuleConfig::get_layer(Layer *layer) const {
    for (auto config : get_child_array("layers"))
        if (config->get("structure") == layer->structure->name and
            config->get("layer") == layer->name)
            return config;
}

Module::Module(LayerList layers, ModuleConfig *config)
        : layers(layers), config(config), curr_iteration(0) {
    for (auto layer : layers) {
        output_types[layer] = Attributes::get_output_type(layer);
        auto layer_config = config->get_layer(layer);

        if (layer_config->get_bool("input", false))
            set_io_type(layer, get_io_type(layer) | INPUT);

        if (layer_config->get_bool("expected", false))
            set_io_type(layer, get_io_type(layer) | EXPECTED);

        if (layer_config->get_bool("output", false))
            set_io_type(layer, get_io_type(layer) | OUTPUT);
    }

    this->verbose = config->get_bool("verbose", false);
    this->start_delay = config->get_int("delay", 0);
    this->cutoff = config->get_int("cutoff", 0);
    this->rate = config->get_int("rate", 1);

    if (this->start_delay < 0)
        LOG_ERROR("Error in Module : " + config->get("type") + "\n"
            " Module start delay must be >= 0!");

    if (this->cutoff < 0)
        LOG_ERROR("Error in Module : " + config->get("type") + "\n"
            " Module cutoff must be >= 0!");

    if (this->rate < 1)
        LOG_ERROR("Error in Module : " + config->get("type") + "\n"
            " Module rate must be >= 1!");
}

void Module::feed_input(Buffer *buffer) {
    if (start_delay <= 0) {
        if (cutoff == 0 or curr_iteration < cutoff) {
            if (curr_iteration % rate == 0)
                feed_input_impl(buffer);
        } else if (curr_iteration == cutoff) {
            for (auto layer : layers)
                if (get_io_type(layer) & INPUT)
                    fSet(buffer->get_input(layer), layer->size, 0.0);
        }
    }
}

void Module::feed_expected(Buffer *buffer) {
    if (start_delay <= 0) {
        if (cutoff == 0 or curr_iteration < cutoff) {
            if (curr_iteration % rate == 0)
                feed_expected_impl(buffer);
        } else if (curr_iteration == cutoff) {
            for (auto layer : layers)
                if (get_io_type(layer) & EXPECTED)
                    fSet((float*)buffer->get_expected(layer).get(), layer->size, 0.0);
        }
    }
}

void Module::report_output(Buffer *buffer) {
    if (start_delay <= 0
            and (cutoff == 0 or curr_iteration < cutoff)
            and (curr_iteration % rate == 0))
        report_output_impl(buffer);
}

void Module::cycle() {
    if (start_delay <= 0) {
        if (cutoff == 0 or curr_iteration < cutoff) {
            ++curr_iteration;
            cycle_impl();
        } else if (cutoff != 0 and curr_iteration == cutoff) {
            ++curr_iteration;
        }
    } else --start_delay;
}

IOTypeMask Module::get_io_type(Layer *layer) const {
    try {
        return io_types.at(layer);
    } catch (std::out_of_range) {
        return 0;
    }
}

/*
 * Checks if two modules are coactive
 * A layer can have two input or expected modules if and only if they are not
 *   active at the same time due to start delays and cutoff iterations
 * This method is used by the Engine to check for this
 */
bool Module::is_coactive(Module* other) const {
    int this_end = this->start_delay + this->cutoff;
    int other_end = other->start_delay + other->cutoff;

    return this->start_delay < other_end and this_end > other->start_delay;
}

Module* Module::build_module(Network *network, ModuleConfig *config) {
    // Check type
    auto type = config->get_type();
    auto bank = Module::get_module_bank();
    if (bank->modules.count(type) == 0)
        LOG_ERROR(
            "Unrecognized module: " + type + "!");

    // Extract layers
    LayerList layers;
    for (auto layer_conf : config->get_layers())
        layers.push_back(
            network->get_structure(layer_conf->get("structure"))
                   ->get_layer(layer_conf->get("layer")));

    // Ensure there are layers in the set
    if (layers.size() == 0) {
        LOG_WARNING(
            "Attempted to build " + type + " module with 0 layers!");
        return nullptr;
    } else {
        // Build using layers and config
        return bank->build_pointers.at(type)(layers, config);
    }
}

Module::ModuleBank* Module::get_module_bank() {
    static Module::ModuleBank* bank = new ModuleBank();
    return bank;
}

int Module::register_module(std::string module_type,
        MODULE_BUILD_PTR build_ptr) {
    auto bank = Module::get_module_bank();
    if (bank->modules.count(module_type) == 1)
        LOG_ERROR(
            "Duplicate module type: " + module_type + "!");
    bank->modules.insert(module_type);
    bank->build_pointers[module_type] = build_ptr;

    // Return index as an identifier
    return bank->modules.size() - 1;
}

void Module::enforce_specified_io_type(std::string type) {
    for (auto layer : layers)
        if (get_io_type(layer) == 0)
            LOG_ERROR(
                type + " module requires specified IO type!");
}

void Module::enforce_unique_io_type(std::string type) {
    for (auto layer : layers)
        if (std::bitset<sizeof(IOTypeMask)>(get_io_type(layer)).count() > 0)
            LOG_ERROR(
                type + " module requires unique IO type!");
}

void Module::enforce_single_io_type(std::string type) {
    IOTypeMask or_type = 0;
    for (auto layer : layers)
        or_type |= get_io_type(layer);

    if (std::bitset<sizeof(IOTypeMask)>(or_type).count() > 0)
        LOG_ERROR(
            type + " module requires single IO type!");
}

void Module::enforce_single_layer(std::string type) {
    if (layers.size() > 1)
        LOG_ERROR(
            type + " module only supports a single layer!");
}

void Module::enforce_equal_layer_sizes(std::string type) {
    if (not check_equal_sizes(layers))
        LOG_ERROR(
            "Layers in " + type + " module must be of equal sizes!");
}

void Module::set_default_io_type(IOTypeMask io_type) {
    for (auto layer : layers)
        if (get_io_type(layer) == 0)
            set_io_type(io_type);
}

void Module::set_io_type(IOTypeMask io_type) {
    for (auto layer : layers)
        io_types[layer] = io_type;
}

void Module::set_io_type(Layer *layer, IOTypeMask io_type) {
    io_types[layer] = io_type;
}

OutputType Module::get_output_type(Layer *layer) {
    try {
        return output_types.at(layer);
    } catch (std::out_of_range) {
        LOG_ERROR(
            "Attempted to retrieve output type from Module for "
            "unrepresented layer: " + layer->str());
    }
}
