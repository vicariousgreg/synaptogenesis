#include <algorithm>
#include <iostream>

#include "io/module/dsst_output_module.h"
#include "util/error_manager.h"
#include "util/tools.h"

REGISTER_MODULE(DSSTOutputModule, "dsst_output", OUTPUT);

DSSTOutputModule::DSSTOutputModule(Layer *layer, ModuleConfig *config)
        : Module(layer) {
    dsst = DSST::get_instance(true);
    if (not dsst->add_output_layer(layer, config->get_property("params")))
        ErrorManager::get_instance()->log_error(
            "Failed to add layer to DSST!");
}

void DSSTOutputModule::report_output(Buffer *buffer, OutputType output_type) {
    Output* output = buffer->get_output(this->layer);

    /*
    int index = 0;
    std::string line;
    do {
        std::cout << "Enter a symbol index (1-9):";
        std::getline(std::cin, line);
        if (line.size() > 0) index = line.at(0) - '0';
        if (index >= 1 and index <= 9) break;
    } while (true);
    std::cout << "Index " << std::to_string(index) << " entered..." << std::endl;
    dsst->input_symbol(index);

    if (fRand() < 0.1)
        dsst->input_symbol(iRand(1,9));
    */
}
