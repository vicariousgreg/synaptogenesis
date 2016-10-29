#include "visualizer.h"
#include "gui.h"

#include <iostream>

Visualizer::Visualizer(Buffer *buffer) : buffer(buffer) {
    this->gui = new GUI();
}

Visualizer::~Visualizer() {
    delete this->gui;
}

void Visualizer::add_layer(Layer *layer, bool input, bool output) {
}

void Visualizer::launch() {
    this->gui->launch();
}

void Visualizer::update() {
    std::cout << "UPDATE\n";
}
