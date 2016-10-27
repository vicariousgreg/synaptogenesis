#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

#include "io/environment.h"
#include "io/module.h"

Environment::Environment(Model *model, Buffer *buffer) : buffer(buffer) {
    // Extract modules
    for (int i = 0; i < model->all_layers.size(); ++i) {
        Module *input_module = model->all_layers[i]->input_module;
        Module *output_module = model->all_layers[i]->output_module;
        if (input_module != NULL)
            this->input_modules.push_back(input_module);
        if (output_module != NULL)
            this->output_modules.push_back(output_module);
    }
}

Environment::~Environment() {
    for (int i = 0; i < this->input_modules.size(); ++i)
        delete this->input_modules[i];
    for (int i = 0; i < this->output_modules.size(); ++i)
        delete this->output_modules[i];
    unlink(this->fifo_name);
    close(this->fifo_fd);
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

#include <iostream>

void Environment::ui_init() {
    // Create FIFO for UI interaction
    mkfifo(this->fifo_name, 0666);

    // Launch UI python process
    std::string head = "python src/pcnn_ui.py ";
    std::string tail = " &";
    std::string command = head + this->fifo_name + tail;
    system(command.c_str());
    this->fifo_fd = open(this->fifo_name, O_WRONLY);

    /* Send preliminary information */
    // Number of output modules
    int num_output_modules = this->output_modules.size();
    write(this->fifo_fd, &num_output_modules, sizeof(int));

    // Layer information
    for (int i = 0; i < this->output_modules.size(); ++i) {
        Layer *layer = this->output_modules[i]->layer;
        write(this->fifo_fd, &layer->output_index, sizeof(int));
        write(this->fifo_fd, &layer->rows, sizeof(int));
        write(this->fifo_fd, &layer->columns, sizeof(int));
    }
}

void Environment::ui_update() {
    //std::cout << "C++\n";
    write(this->fifo_fd, buffer->get_output(), buffer->output_size * sizeof(Output));
}
