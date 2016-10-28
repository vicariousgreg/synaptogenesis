#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>

#include "io/visualizer.h"

#include <iostream>

Visualizer::Visualizer(Buffer *buffer) : buffer(buffer) { }

Visualizer::~Visualizer() {
    unlink(this->fifo_name);
    close(this->fifo_fd);
}

void Visualizer::add_layer(Layer *layer, bool input, bool output) {
    std::cout << input << std::endl;
    std::cout << output << std::endl;
    this->layer_infos.push_back(LayerInfo(layer, input, output));
}

void Visualizer::ui_init() {
    // Create FIFO for UI interaction
    mkfifo(this->fifo_name, 0666);

    // Launch UI python process
    std::string head = "python src/pcnn_ui.py ";
    std::string out_type;
    switch (this->buffer->output_type) {
        case FLOAT:
            out_type = " float ";
            break;
        case INT:
            out_type = " int ";
            break;
        case BIT:
            out_type = " bit ";
            break;
    }
    std::string tail = " &";
    std::string command = head + this->fifo_name + out_type + tail;
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
    std::cout << "C++\n";
    for (int i = 0; i < this->layer_infos.size(); ++i) {
        LayerInfo info = this->layer_infos[i];
        if (info.is_output != 0)
            write(this->fifo_fd,
                buffer->get_output() + info.output_index,
                info.size * sizeof(Output));
    }
}
