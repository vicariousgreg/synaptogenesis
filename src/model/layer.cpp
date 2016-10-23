#include "io/module.h"

Layer::Layer(std::string name, int start_index, int rows, int columns, std::string params) :
        name(name),
        index(start_index),
        rows(rows),
        columns(columns),
        size(rows * columns),
        params(params),
        type(INTERNAL),
        input_index(0),
        output_index(0),
        input_module(NULL),
        output_module(NULL) {}

void Layer::add_module(Module *module) {
    IOType new_type = module->get_type();

    switch (new_type) {
        case INPUT:
            if (this->input_module != NULL)
                throw "Layer cannot have more than one input module!";
            this->input_module = module;
            break;
        case OUTPUT:
            if (this->output_module != NULL)
                throw "Layer cannot have more than one output module!";
            this->output_module = module;
            break;
        case INPUT_OUTPUT:
            if (this->input_module != NULL or this->output_module != NULL)
                throw "Layer cannot have more than one input/output module!";
            this->input_module = module;
            this->output_module = module;
            break;
        default:
            throw "Unrecognized module type!";
    }

    if (this->input_module != NULL and this->output_module != NULL)
        this->type = INPUT_OUTPUT;
    else if (this->input_module != NULL)
        this->type = INPUT;
    else if (this->output_module != NULL)
        this->type = OUTPUT;
    else  // This shouldn't happen, but here it is anyway
        this->type = INTERNAL;
}
