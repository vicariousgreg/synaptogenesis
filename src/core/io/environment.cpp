#include <vector>

#include "io/environment.h"
#include "io/module/module.h"
#include "io/module/visualizer_input_module.h"
#include "io/module/visualizer_output_module.h"

Environment::Environment(Model *model, Buffer *buffer)
        : buffer(buffer),
          visualizer(NULL) {
    // Extract modules
    for (auto& layer : model->get_layers()) {
        bool visualizer_input = false;
        bool visualizer_output = false;

        // Add input module
        // If visualizer input module, set flag
        Module *input_module = layer->get_input_module();
        if (input_module != NULL) {
            this->input_modules.push_back(input_module);
            visualizer_input =
                dynamic_cast<VisualizerInputModule*>(input_module) != NULL;
        }

        // Add output modules
        // If visualizer output module is found, set flag
        auto output_modules = layer->get_output_modules();
        for (int j = 0; j < output_modules.size(); ++j) {
            Module *output_module = output_modules[j];
            this->output_modules.push_back(output_module);

            visualizer_output |=
                dynamic_cast<VisualizerOutputModule*>(output_module) != NULL;
        }

        if (visualizer_input or visualizer_output) {
            if (visualizer == NULL)
                visualizer = new Visualizer(buffer);
            visualizer->add_layer(layer, visualizer_input, visualizer_output);
        }

    }
}

Environment::~Environment() {
    for (auto& module : this->input_modules) delete module;
    for (auto& module : this->output_modules) delete module;
    if (visualizer != NULL)
        delete visualizer;
}

void Environment::step_input() {
    // Run input modules
    for (int i = 0 ; i < this->input_modules.size(); ++i)
        this->input_modules[i]->feed_input(buffer);
}

void Environment::step_output() {
    // Run output modules
    // If no module, skip layer
    for (int i = 0 ; i < this->output_modules.size(); ++i)
        this->output_modules[i]->report_output(buffer);
}

void Environment::ui_launch() {
    if (visualizer != NULL)
        visualizer->launch();
}

void Environment::ui_update() {
    if (visualizer != NULL)
        visualizer->update();
}
