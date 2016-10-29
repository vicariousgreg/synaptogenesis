#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>

#include "model/model_builder.h"

ModelBuilder::ModelBuilder() { }

ModelBuilder::~ModelBuilder() {
}

void ModelBuilder::load(std::string path) {
    // Create FIFO for UI interaction
    mkfifo(this->fifo_name, 0666);

    // Launch UI python process
    std::string command = "python ";
    command.append(this->script);
    command.append(" ");
    command.append(this->fifo_name);
    command.append(" ");
    command.append(path);
    command.append(" &");
    system(command.c_str());

    this->fifo_fd = open(this->fifo_name, O_RDONLY);
    unlink(this->fifo_name);
    close(this->fifo_fd);
}

/*
void ModelBuilder::ui_update() {
    //std::cout << "C++\n";
    for (int i = 0; i < this->layer_infos.size(); ++i) {
        LayerInfo info = this->layer_infos[i];
        if (info.is_output != 0)
            write(this->fifo_fd,
                buffer->get_output() + info.output_index,
                info.size * sizeof(Output));
    }
}
*/
