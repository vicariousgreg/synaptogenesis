#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

#include "io/visualizer.h"

#include <iostream>

Visualizer::Visualizer(Buffer *buffer) : buffer(buffer) { }

Visualizer::~Visualizer() {
    //unlink(this->fifo_name);
    //close(this->fifo_fd);
}

void Visualizer::add_layer(Layer *layer, bool input, bool output) {
    this->layer_infos.push_back(LayerInfo(layer, input, output));
}

void Visualizer::ui_init() {
    // Create FIFO for UI interaction
    mkfifo(this->fifo_name, 0666);

    // Launch UI python process
    std::string command = "python ";
    command.append(this->ui_script);
    command.append(" ");
    command.append(this->fifo_name);
    switch (this->buffer->output_type) {
        case FLOAT:
            command.append(" float ");
            break;
        case INT:
            command.append(" int ");
            break;
        case BIT:
            command.append(" bit ");
            break;
    }
    command.append(" &");
    system(command.c_str());
    this->fifo_fd = open(this->fifo_name, O_WRONLY);

    /* Send preliminary information */
    int num_layers = this->layer_infos.size();
    write(this->fifo_fd, &num_layers, sizeof(int));

    // Layer information
    for (int i = 0; i < this->layer_infos.size(); ++i) {
        LayerInfo info = this->layer_infos[i];
        write(this->fifo_fd, &info.input_index, sizeof(int));
        write(this->fifo_fd, &info.output_index, sizeof(int));
        write(this->fifo_fd, &info.rows, sizeof(int));
        write(this->fifo_fd, &info.columns, sizeof(int));
        write(this->fifo_fd, &info.is_input, sizeof(int));
        write(this->fifo_fd, &info.is_output, sizeof(int));
    }
}

void Visualizer::ui_update() {
    //std::cout << "C++\n";
    for (int i = 0; i < this->layer_infos.size(); ++i) {
        LayerInfo info = this->layer_infos[i];
        if (info.is_output != 0)
            write(this->fifo_fd,
                buffer->get_output() + info.output_index,
                info.size * sizeof(Output));
    }
}
