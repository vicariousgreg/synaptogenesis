#include <sstream>

#include "model/connection_config.h"
#include "util/error_manager.h"

ConnectionConfig::ConnectionConfig(
    bool plastic, int delay, float max_weight,
    ConnectionType type, Opcode opcode,
    std::string connection_params, WeightConfig* weight_config)
        : plastic(plastic),
          delay(delay),
          max_weight(max_weight),
          type(type),
          opcode(opcode),
          connection_params(connection_params),
          weight_config(weight_config) { }

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

ArborizedConfig::ArborizedConfig( int field_size, int stride, int offset)
    : ArborizedConfig(field_size, field_size, stride, stride, offset, offset) { }

ArborizedConfig::operator std::string() const {
    std::ostringstream oss;
    oss << row_field_size << " ";
    oss << column_field_size << " ";
    oss << row_stride << " ";
    oss << column_stride << " ";
    oss << row_offset << " ";
    oss << column_offset << " ";
    return oss.str();
}

ArborizedConfig ArborizedConfig::decode(std::string params) {
    int row_field_size, column_field_size;
    int row_stride, column_stride;
    int row_offset, column_offset;

    std::stringstream stream(params);

    // Extract field size
    if (stream.eof())
        ErrorManager::get_instance()->log_error(
            "Row field size for arborized connection not specified!");
    stream >> row_field_size;
    if (stream.eof())
        ErrorManager::get_instance()->log_error(
            "Column field size for arborized connection not specified!");
    stream >> column_field_size;

    // Extract stride
    if (stream.eof())
        ErrorManager::get_instance()->log_error(
            "Row stride for arborized connection not specified!");
    stream >> row_stride;
    if (stream.eof())
        ErrorManager::get_instance()->log_error(
            "Column stride for arborized connection not specified!");
    stream >> column_stride;

    // Extract offset
    if (stream.eof())
        ErrorManager::get_instance()->log_error(
            "Row offset for arborized connection not specified!");
    stream >> row_offset;
    if (stream.eof())
        ErrorManager::get_instance()->log_error(
            "Column offset for arborized connection not specified!");
    stream >> column_offset;

    return ArborizedConfig(row_field_size, column_field_size,
        row_stride, column_stride, row_offset, column_offset);
}
