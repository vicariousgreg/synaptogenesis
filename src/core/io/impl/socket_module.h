#ifndef socket_module_h
#define socket_module_h

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "io/module.h"

class SocketModule : public Module {
    public:
        SocketModule(LayerList layers, ModuleConfig *config);

        void feed_input_impl(Buffer *buffer);
        void feed_expected_impl(Buffer *buffer);
        void report_output_impl(Buffer *buffer);

    protected:
        bool stream_input, stream_expected, stream_output;
        bool single_layer;
        int server, client;
        struct sockaddr_in myaddr;
        Pointer<float> local_buffer;
        int buffer_bytes;

    MODULE_MEMBERS
};

#endif
