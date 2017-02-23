#include "io/environment.h"
#include "io/module/module.h"
#include "io/module/visualizer_input_module.h"
#include "io/module/visualizer_output_module.h"
#include "state/state.h"
#include "visualizer.h"

Environment::Environment(State *state)
        : state(state), visualizer(NULL) {
    // Create buffers
    for (auto& structure : state->model->get_structures())
        this->buffers[structure] = new Buffer(structure);

    // Extract modules
    for (auto& structure : state->model->get_structures()) {
        for (auto& layer : structure->get_layers()) {
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
            for (auto& output_module : output_modules) {
                this->output_modules.push_back(output_module);
                visualizer_output |=
                    dynamic_cast<VisualizerOutputModule*>(output_module) != NULL;
            }

            if (visualizer_input or visualizer_output) {
                if (visualizer == NULL)
                    visualizer = new Visualizer(this);
                visualizer->add_layer(layer, visualizer_input, visualizer_output);
            }
        }
    }
}

Environment::~Environment() {
    for (auto buffer : buffers) delete buffer.second;
    for (auto& module : this->input_modules) delete module;
    for (auto& module : this->output_modules) delete module;
    if (visualizer != NULL) delete visualizer;
}

Buffer* Environment::get_buffer(Structure *structure) {
    return buffers.at(structure);
}

OutputType Environment::get_output_type(Structure *structure) {
    return state->get_output_type(structure);
}

void Environment::step_input() {
    for (auto& module : this->input_modules)
        module->feed_input(buffers.at(module->layer->structure));
}

void Environment::step_output() {
    for (auto& module : this->output_modules)
        module->report_output(
            buffers.at(module->layer->structure),
            state->get_output_type(module->layer->structure));
}

void Environment::ui_launch() {
    if (visualizer != NULL)
        visualizer->launch();
}

void Environment::ui_update() {
    if (visualizer != NULL)
        visualizer->update();
}
