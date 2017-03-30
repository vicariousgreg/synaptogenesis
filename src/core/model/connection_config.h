#ifndef connection_config_h
#define connection_config_h

#include <string>
#include "util/constants.h"

class ConnectionConfig {
    public:
        ConnectionConfig(
            bool plastic,
            int delay,
            float max_weight,
            ConnectionType type,
            Opcode opcode,
            std::string connection_params,
            std::string init_params);

        bool plastic;
        int delay;
        float max_weight;
        ConnectionType type;
        std::string connection_params;
        std::string init_params;
        Opcode opcode;
};

class ArborizedConfig {
    public:
        ArborizedConfig(
            int row_field_size, int column_field_size,
            int row_stride, int column_stride);

        ArborizedConfig( int field_size, int stride);

        std::string encode();

        static ArborizedConfig decode(std::string params);

        const int row_field_size, column_field_size;
        const int row_stride, column_stride;
};

#endif
