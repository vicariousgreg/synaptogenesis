#ifndef visualizer_h
#define visualizer_h

#include "model/layer.h"
#include "io/buffer.h"

class Visualizer {
    public:
        Visualizer(Buffer *buffer);
        virtual ~Visualizer();

        void add_layer(Layer *layer, bool input, bool output);
        void ui_init();
        void ui_launch();
        void ui_update();

    private:
        Buffer *buffer;
};

#endif
