#ifndef model_builder_h
#define model_builder_h

#include "model/model.h"

class ModelBuilder {
    public:
        ModelBuilder();
        virtual ~ModelBuilder();

        void load(std::string path);

        void ui_init();

    private:
        const char *fifo_name = "/tmp/synaptogenesis_fifo";
        const char *script = "src/core/python/model_builder.py";
        int fifo_fd;
};

#endif
