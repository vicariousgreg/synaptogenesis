#ifndef module_h
#define module_h

#include <string>

#include "model/layer.h"
#include "io/buffer.h"

class Module {
    public:
        Module(Layer *layer) : layer(layer) { }
        virtual ~Module() {}

        /* Override to implement input and output functionality.
         * If unused, do not override */
        virtual void feed_input(Buffer *buffer) { }
        virtual void report_output(Buffer *buffer) { }

        /* Override to indicate IO type
         * This is used by the environment to determine which hooks to call
         */
        virtual IOType get_type() = 0;

        Layer *layer;
};

Module* build_module(Layer *layer, std::string type, std::string params);

#endif
