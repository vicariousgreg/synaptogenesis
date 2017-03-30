#include <sstream>

#include "model/connection_config.h"
#include "util/error_manager.h"

ArborizedConfig::ArborizedConfig(
    int row_field_size, int column_field_size,
    int row_stride, int column_stride,
    int row_offset, int column_offset)
        : row_field_size(row_field_size),
          column_field_size(column_field_size),
          row_stride(row_stride),
          column_stride(column_stride),
          row_offset(row_offset),
          column_offset(column_offset) { }

ArborizedConfig::ArborizedConfig(int field_size, int stride, int offset)
    : ArborizedConfig(field_size, field_size, stride, stride, offset, offset) { }

ConnectionConfig::ConnectionConfig(
    bool plastic, int delay, float max_weight,
    ConnectionType type, Opcode opcode,
    WeightConfig* weight_config)
        : plastic(plastic),
          delay(delay),
          max_weight(max_weight),
          type(type),
          opcode(opcode),
          weight_config(weight_config) { }

ConnectionConfig::ConnectionConfig(
    bool plastic, int delay, float max_weight,
    ConnectionType type, Opcode opcode,
    WeightConfig* weight_config,
    ArborizedConfig arborized_config)
        : plastic(plastic),
          delay(delay),
          max_weight(max_weight),
          type(type),
          opcode(opcode),
          weight_config(weight_config),
          arborized_config(arborized_config) { }
