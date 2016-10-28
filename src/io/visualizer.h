#ifndef visualizer_h
#define visualizer_h

#include "model/layer.h"
#include "io/buffer.h"

class LayerInfo {
    public:
        LayerInfo(Layer *layer, bool is_input, bool is_output) 
            : input_index(layer->input_index),
              output_index(layer->output_index),
              rows(layer->rows),
              columns(layer->columns),
              size(layer->rows * layer->columns),
              is_input(is_input),
              is_output(is_output) {
        }

        int input_index, output_index;
        int rows, columns, size;
        int is_input, is_output;
};

class Visualizer {
    public:
        Visualizer(Buffer *buffer);
        virtual ~Visualizer();

        void add_layer(Layer *layer, bool input, bool output);
        void ui_init();
        void ui_update();

    private:
        Buffer *buffer;
        std::vector<LayerInfo> layer_infos;

        const char *fifo_name = "/tmp/pcnn_fifo";
        const char *ui_script = "src/ui/ui_main.py";
        int fifo_fd;
};

#endif
