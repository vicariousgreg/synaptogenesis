#include "socket_module.h"

static char ping_char = ' ';

REGISTER_MODULE(SocketModule, "socket");

SocketModule::SocketModule(LayerList layers, ModuleConfig *config)
        : Module(layers, config), stream_input(false), stream_expected(false) {
    enforce_equal_layer_sizes("socket");
    this->single_layer = layers.size() == 1;

    for (auto layer : layers) {
        auto layer_config = config->get_layer(layer);

        if (layer_config->get_bool("input", false)) {
            set_io_type(layer, get_io_type(layer) | INPUT);
            this->stream_input = true;
        }

        if (layer_config->get_bool("expected", false)) {
            if (get_io_type(layer) != 0)
                LOG_ERROR("Found multiple IO types for "
                    + layer->str() + " in SocketModule!");
            set_io_type(layer, get_io_type(layer) | EXPECTED);
            this->stream_expected = true;
        }

        if (layer_config->get_bool("output", false)) {
            if (get_io_type(layer) != 0)
                LOG_ERROR("Found multiple IO types for "
                    + layer->str() + " in SocketModule!");
            else set_io_type(layer, get_io_type(layer) | OUTPUT);
        }

        // Log error if unspecified type
        if (get_io_type(layer) == 0)
            LOG_ERROR("Unspecified type for layer "
                + layer->str() + " in SocketModule!");
    }

    // Port and IP address
    int port = config->get_int("port", 11111);
    std::string ip = config->get("ip", "192.168.0.180");

    // Create address structure
    myaddr.sin_family = AF_INET;
    myaddr.sin_port = htons(port);
    inet_aton(ip.c_str(), &myaddr.sin_addr);

    // Open and configure socket
    this->server = socket(PF_INET, SOCK_STREAM, 0);
    if (this->server < 0)
        LOG_ERROR("Failed to open socket on "
            + ip + " " + std::to_string(port));

    int yes = 1;
    if (setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) == -1)
        LOG_ERROR("Failed to set socket options for socket on "
            + ip + " " + std::to_string(port));

    // Bind to address
    if (bind(this->server, (struct sockaddr*)&myaddr, sizeof(myaddr)) < 0)
        LOG_ERROR("Failed to bind socket on "
            + ip + " " + std::to_string(port));

    // Connect to client
    socklen_t size;
    listen(server, 1);
    this->client = accept(server, (struct sockaddr *) &myaddr, &size);

    if (this->client < 0)
        LOG_ERROR("Failed to accept socket connection to client on "
            + ip + " " + std::to_string(port));

    // Create buffer if multiple layers
    if (not single_layer)
        local_buffer = Pointer<float>(layers[0]->size, 0.0);
    buffer_bytes = layers[0]->size * sizeof(float);
}

void SocketModule::feed_input_impl(Buffer *buffer) {
    if (single_layer) {
        // If there's only one layer, stream directly into layer buffer
        auto layer = layers[0];
        if (get_io_type(layer) & INPUT) {
            send(client, &ping_char, 1, 0);
            recv(client, buffer->get_input(layer).get(), buffer_bytes, 0);
        }
    } else {
        // Otherwise, stream into local buffer and copy
        send(client, &ping_char, 1, 0);
        recv(client, local_buffer.get(), buffer_bytes, 0);

        for (auto layer : layers)
            if (get_io_type(layer) & INPUT)
                buffer->set_input(layer, this->local_buffer);
    }
}

void SocketModule::feed_expected_impl(Buffer *buffer) {
    if (single_layer) {
        // If there's only one layer, stream directly into layer buffer
        auto layer = layers[0];
        if (get_io_type(layer) & EXPECTED) {
            send(client, &ping_char, 1, 0);
            recv(client, buffer->get_expected(layer).get(), buffer_bytes, 0);
        }
    } else {
        // Otherwise, stream into local buffer and copy
        send(client, &ping_char, 1, 0);
        recv(client, local_buffer.get(), buffer_bytes, 0);

        for (auto layer : layers)
            if (get_io_type(layer) & EXPECTED)
                buffer->set_expected(layer, this->local_buffer.cast<Output>());
    }
}

void SocketModule::report_output_impl(Buffer *buffer) {
    for (auto layer : layers) {
        if (get_io_type(layer) & OUTPUT) {
            send(client, &ping_char, 1, 0);
            send(client, buffer->get_output(layer).get(), buffer_bytes, 0);
        }
    }
}
