#ifndef visualizer_h
#define visualizer_h

#include "model/layer.h"
#include "io/buffer.h"
#include "layer_info.h"

class GUI;

class Visualizer {
    public:
        Visualizer(Buffer *buffer);
        virtual ~Visualizer();

        void add_layer(Layer *layer, bool input, bool output);
        void launch();
        void update();

    private:
        Buffer *buffer;
        GUI *gui;
};

#endif
