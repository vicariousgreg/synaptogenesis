#include "io/impl/socket_module.h"

static char ping_char = ' ';

REGISTER_MODULE(SocketModule, "socket");

SocketModule::SocketModule(LayerList layers, ModuleConfig *config)
        : Module(layers, config),
          stream_input(false),
          stream_expected(false),
          stream_output(false) {
    enforce_equal_layer_sizes("socket");
    enforce_specified_io_type("socket");
    enforce_unique_io_type("socket");
    enforce_single_io_type("socket");

    auto io_type = get_io_type(layers[0]);
    this->stream_input = io_type == INPUT;
    this->stream_expected = io_type == EXPECTED;
    this->stream_output = io_type == OUTPUT;

    // Ensure one IO type
    if (stream_input + stream_expected + stream_output > 1)
        LOG_ERROR("Cannot use same SocketModule for more than one IO type!");

    // Create buffer if multiple layers
    this->single_layer = layers.size() == 1;
    if (not single_layer)
        local_buffer = Pointer<float>(layers[0]->size, 0.0);
    buffer_bytes = layers[0]->size * sizeof(float);

    // Port and IP address
    int port = config->get_int("port", 11111);
    std::string ip = config->get("ip", "192.168.0.180");
    std::string socket_string =
        "[ip = " + ip + " : port = " + std::to_string(port) + "]";

    // Create address structure
    myaddr.sin_family = AF_INET;
    myaddr.sin_port = htons(port);
    inet_aton(ip.c_str(), &myaddr.sin_addr);

    // Open and configure socket
    this->server = socket(PF_INET, SOCK_STREAM, 0);
    if (this->server < 0)
        LOG_ERROR("Failed to open socket on " + socket_string);

    int yes = 1;
    if (setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) == -1)
        LOG_ERROR("Failed to set socket options for socket on "
            + socket_string);

    // Bind to address
    if (bind(this->server, (struct sockaddr*)&myaddr, sizeof(myaddr)) < 0)
        LOG_ERROR("Failed to bind socket on " + socket_string);

    // Connect to client
    socklen_t size;
    listen(server, 1);
    this->client = accept(server, (struct sockaddr *) &myaddr, &size);

    if (this->client < 0)
        LOG_ERROR("Failed to accept socket connection to client on "
            + socket_string);
}

static void get_mesg(int client, void* ptr, size_t size) {
    send(client, &ping_char, 1, 0);

    char *p = (char*) ptr;
    ssize_t n;
    while (size > 0 && (n = recv(client,p,size,0)) > 0) {
        p += n;
        size -= (size_t)n;
    }

    if ( size > 0 || n < 0 )
        LOG_ERROR("Socket error!");
}

static void send_mesg(int client, void* ptr, size_t size) {
    send(client, &ping_char, 1, 0);

    char *p = (char*) ptr;
    ssize_t n;
    while (size > 0 && (n = send(client,p,size,0)) > 0) {
        p += n;
        size -= (size_t)n;
    }

    if (size > 0 || n < 0) LOG_ERROR("Socket error!");
}

void SocketModule::feed_input_impl(Buffer *buffer) {
    if (not stream_input) return;

    if (single_layer) {
        // If there's only one layer, stream directly into layer buffer
        get_mesg(client, buffer->get_input(layers[0]).get(), buffer_bytes);
    } else {
        // Otherwise, stream into local buffer and copy
        get_mesg(client, local_buffer.get(), buffer_bytes);

        for (auto layer : layers)
            if (get_io_type(layer) & INPUT)
                buffer->set_input(layer, this->local_buffer);
    }
}

void SocketModule::feed_expected_impl(Buffer *buffer) {
    if (not stream_expected) return;

    if (single_layer) {
        // If there's only one layer, stream directly into layer buffer
        get_mesg(client, buffer->get_expected(layers[0]).get(), buffer_bytes);
    } else {
        // Otherwise, stream into local buffer and copy
        get_mesg(client, local_buffer.get(), buffer_bytes);

        for (auto layer : layers)
            if (get_io_type(layer) & EXPECTED)
                buffer->set_expected(layer, this->local_buffer.cast<Output>());
    }
}

void SocketModule::report_output_impl(Buffer *buffer) {
    if (not stream_output) return;

    for (auto layer : layers)
        if (get_io_type(layer) & OUTPUT)
            send_mesg(client, buffer->get_output(layer).get(), buffer_bytes);
}
