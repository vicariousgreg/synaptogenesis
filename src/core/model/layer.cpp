#include "model/layer.h"
#include "io/module/module.h"
#include "util/error_manager.h"

Layer::Layer(Structure* structure, std::string name, int rows, int columns, std::string params) :
        name(name),
        structure(structure),
        start_index(0),
        rows(rows),
        columns(columns),
        size(rows * columns),
        params(params),
        type(INTERNAL),
        input_index(0),
        output_index(0),
        input_module(NULL),
        dendritic_root(new DendriticNode(0, this)) { }

void Layer::add_input_connection(Connection* connection) {
    this->input_connections.push_back(connection);
}

void Layer::add_output_connection(Connection* connection) {
    this->output_connections.push_back(connection);
}

void Layer::add_to_root(Connection* connection) {
    this->dendritic_root->add_child(connection);
}

void Layer::add_module(Module *module) {
    IOType new_type = module->get_type();

    switch (new_type) {
        case INPUT:
            if (this->input_module != NULL)
                ErrorManager::get_instance()->log_error(
                    "Layer cannot have more than one input module!");
            this->input_module = module;
            break;
        case OUTPUT:
            this->output_modules.push_back(module);
            break;
        case INPUT_OUTPUT:
            if (this->input_module != NULL)
                ErrorManager::get_instance()->log_error(
                    "Layer cannot have more than one input module!");
            this->input_module = module;
            this->output_modules.push_back(module);
            break;
        default:
            ErrorManager::get_instance()->log_error(
                "Unrecognized module type!");
    }

    if (this->input_module != NULL and this->output_modules.size() > 0)
        this->type = INPUT_OUTPUT;
    else if (this->input_module != NULL)
        this->type = INPUT;
    else if (this->output_modules.size() > 0)
        this->type = OUTPUT;
    else  // This shouldn't happen, but here it is anyway
        this->type = INTERNAL;
}
