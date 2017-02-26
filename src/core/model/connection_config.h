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
            std::string params)
                : plastic(plastic),
                  delay(delay),
                  max_weight(max_weight),
                  type(type),
                  opcode(opcode),
                  params(params) { }

        bool plastic;
        int delay;
        float max_weight;
        ConnectionType type;
        std::string params;
        Opcode opcode;
};

#endif
