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

FullyConnectedConfig::FullyConnectedConfig(
    int from_row_start, int from_row_end,
    int from_col_start, int from_col_end,
    int to_row_start, int to_row_end,
    int to_col_start, int to_col_end)
        : from_row_start(from_row_start),
          from_row_end(from_row_end),
          from_row_size(from_row_end - from_row_start),
          from_col_start(from_col_start),
          from_col_end(from_col_end),
          from_col_size(from_col_end - from_col_start),
          from_size(from_row_size * from_col_size),
          to_row_start(to_row_start),
          to_row_end(to_row_end),
          to_row_size(to_row_end - to_row_start),
          to_col_start(to_col_start),
          to_col_end(to_col_end),
          to_col_size(to_col_end - to_col_start),
          to_size(to_row_size * to_col_size),
          total_size(from_size * to_size) { }

ConnectionConfig::ConnectionConfig(
    bool plastic, int delay, float max_weight,
    ConnectionType type, Opcode opcode,
    WeightConfig* weight_config)
        : plastic(plastic),
          delay(delay),
          max_weight(max_weight),
          type(type),
          opcode(opcode),
          weight_config(weight_config),
          arborized_config(nullptr),
          fully_connected_config(nullptr) { }

ConnectionConfig::~ConnectionConfig() {
    delete weight_config;
    if (arborized_config != nullptr) delete arborized_config;
    if (fully_connected_config != nullptr) delete fully_connected_config;
}

int ConnectionConfig::get_expected_rows(int rows) {
    switch (type) {
        case(ONE_TO_ONE):
            return rows;
        case(FULLY_CONNECTED):
            return rows;
        default:
            int row_field_size = arborized_config->row_field_size;
            int row_stride = arborized_config->row_stride;
            switch(type) {
                case(CONVERGENT):
                case(CONVOLUTIONAL):
                    if (row_stride == 0) return rows;
                    return std::max(1,
                        1 + ((rows - row_field_size) / row_stride));
                case(DIVERGENT):
                    return std::max(1,
                        row_field_size + (row_stride * (rows - 1)));
                default:
                    ErrorManager::get_instance()->log_error(
                        "Invalid call to get_expected_rows!");
            }
    }
}

int ConnectionConfig::get_expected_columns(int columns) {
    switch (type) {
        case(ONE_TO_ONE):
            return columns;
        case(FULLY_CONNECTED):
            return columns;
        default:
            int column_field_size = arborized_config->column_field_size;
            int column_stride = arborized_config->column_stride;
            switch(type) {
                case(CONVERGENT):
                case(CONVOLUTIONAL):
                    if (column_stride == 0) return columns;
                    return std::max(1,
                        1 + ((columns - column_field_size) / column_stride));
                case(DIVERGENT):
                    return std::max(1,
                        column_field_size + (column_stride * (columns - 1)));
                default:
                    ErrorManager::get_instance()->log_error(
                        "Invalid call to get_expected_columns!");
            }
    }
}
