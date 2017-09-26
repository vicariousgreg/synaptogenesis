#include <sstream>

#include "network/layer.h"
#include "network/connection.h"
#include "network/connection_config.h"
#include "util/error_manager.h"

ArborizedConfig::ArborizedConfig(
    int row_field_size, int column_field_size,
    int row_stride, int column_stride,
    bool wrap)
        : ArborizedConfig(row_field_size, column_field_size,
                          row_stride, column_stride,
                          -row_field_size/2, -column_field_size/2,
                          wrap) { }

ArborizedConfig::ArborizedConfig(
    int row_field_size, int column_field_size,
    int row_stride, int column_stride,
    int row_offset, int column_offset,
    bool wrap)
        : row_field_size(row_field_size),
          column_field_size(column_field_size),
          row_stride(row_stride),
          column_stride(column_stride),
          row_offset(row_offset),
          column_offset(column_offset),
          wrap(wrap) { }

ArborizedConfig::ArborizedConfig(int field_size, int stride, bool wrap)
    : ArborizedConfig(field_size, field_size,
                      stride, stride,
                      -field_size/2, -field_size/2,
                      wrap) { }

ArborizedConfig::ArborizedConfig(int field_size,
        int stride, int offset, bool wrap)
    : ArborizedConfig(field_size, field_size,
                      stride, stride,
                      offset, offset,
                      wrap) { }

std::string ArborizedConfig::str() const {
    return "(" +
        std::to_string(row_field_size) + "-" +
        std::to_string(column_field_size) + "-" +
        std::to_string(row_stride) + "-" +
        std::to_string(column_stride) + "-" +
        std::to_string(row_offset) + "-" +
        std::to_string(column_offset) + "-" +
        std::to_string(wrap) + ")";
}

SubsetConfig::SubsetConfig(
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
          total_size(from_size * to_size) {
        if (from_row_start < 0 or from_col_start < 0
                or to_row_start < 0 or to_col_start < 0)
            ErrorManager::get_instance()->log_error(
                "SubsetConfig Connected Config cannot have"
                " negative start indices!");
        if (from_row_start > from_row_end or from_col_start > from_col_end
                or to_row_start > to_row_end or to_col_start > to_col_end)
            ErrorManager::get_instance()->log_error(
                "SubsetConfig Connected Config cannot have start indices"
                " greater than end indices!");
}

bool SubsetConfig::validate(Connection *conn) {
    Layer *from_layer = conn->from_layer;
    Layer *to_layer = conn->to_layer;
    return
        from_row_end <= from_layer->rows
        and from_col_end <= from_layer->columns
        and to_row_end <= to_layer->rows
        and to_col_end <= to_layer->columns;
}

std::string SubsetConfig::str() const {
    return " (" +
        std::to_string(from_row_start) + "-" +
        std::to_string(from_row_end) + "-" +
        std::to_string(from_col_start) + "-" +
        std::to_string(from_col_end) + "-" +
        std::to_string(to_row_start) + "-" +
        std::to_string(to_row_end) + "-" +
        std::to_string(to_col_start) + "-" +
        std::to_string(to_col_end) + ")";
}

ConnectionConfig::ConnectionConfig(PropertyConfig *config)
    : plastic(config->get("plastic", "true") == "true"),
      delay(std::stoi(config->get("delay", "0"))),
      max_weight(std::stof(config->get("max", "1.0"))),
      type(ConnectionTypes.at(config->get("type", "fully connected"))),
      opcode(Opcodes.at(config->get("opcode", "add"))),
      weight_config(new FlatWeightConfig(1.0)),
      subset_config(nullptr),
      arborized_config(nullptr) {
    for (auto pair : config->get())
        this->set(pair.first, pair.second);
    for (auto pair : config->get_children()) {
        if (pair.first == "weight config") {
            set_weight_config(new WeightConfig(pair.second));
        } else if (pair.first == "subset config") {
            int from_row_start = 0;
            int from_row_end = 0;
            int from_col_start = 0;
            int from_col_end = 0;
            int to_row_start = 0;
            int to_row_end = 0;
            int to_col_start = 0;
            int to_col_end = 0;

            for (auto c_pair : pair.second->get()) {
                if (c_pair.first == "from row start")
                    from_row_start = std::stoi(c_pair.second);
                else if (c_pair.first == "from row end")
                    from_row_end = std::stoi(c_pair.second);
                else if (c_pair.first == "from column start")
                    from_col_start = std::stoi(c_pair.second);
                else if (c_pair.first == "from column end")
                    from_col_end = std::stoi(c_pair.second);
                else if (c_pair.first == "to row start")
                    to_row_start = std::stoi(c_pair.second);
                else if (c_pair.first == "to row end")
                    to_row_end = std::stoi(c_pair.second);
                else if (c_pair.first == "to column start")
                    to_col_start = std::stoi(c_pair.second);
                else if (c_pair.first == "to column end")
                    to_col_end = std::stoi(c_pair.second);
                else
                    ErrorManager::get_instance()->log_error(
                        "Unrecognized subset config property: " + c_pair.first);
            }
            set_subset_config(
                new SubsetConfig(
                    from_row_start, from_row_end,
                    from_col_start, from_col_end,
                    to_row_start, to_row_end,
                    to_col_start, to_col_end));
        } else if (pair.first == "arborized config") {
            int row_field_size = -1;
            int column_field_size = -1;
            int row_stride = 1;
            int column_stride = 1;
            int row_offset = -1;
            int column_offset = -1;
            bool wrap = false;

            for (auto c_pair : pair.second->get()) {
                if (c_pair.first == "row field size")
                    row_field_size = std::stoi(c_pair.second);
                else if (c_pair.first == "column field size")
                    column_field_size = std::stoi(c_pair.second);
                else if (c_pair.first == "field size")
                    row_field_size = column_field_size =
                        std::stoi(c_pair.second);
                else if (c_pair.first == "row stride")
                    row_stride = std::stoi(c_pair.second);
                else if (c_pair.first == "column stride")
                    column_stride = std::stoi(c_pair.second);
                else if (c_pair.first == "stride")
                    row_stride = column_stride = std::stoi(c_pair.second);
                else if (c_pair.first == "row offset")
                    row_offset = std::stoi(c_pair.second);
                else if (c_pair.first == "column offset")
                    column_offset = std::stoi(c_pair.second);
                else if (c_pair.first == "offset")
                    row_offset = column_offset = std::stoi(c_pair.second);
                else if (c_pair.first == "wrap")
                    wrap = c_pair.second == "true";
                else
                    ErrorManager::get_instance()->log_error(
                        "Unrecognized arborized config property: " + c_pair.first);
            }

            if (row_field_size < 0 or column_field_size < 0)
                ErrorManager::get_instance()->log_error(
                    "Unspecified field size for arborized config!");

            if (row_offset < 0 and column_offset < 0) {
                set_arborized_config(
                    new ArborizedConfig(
                        row_field_size, column_field_size,
                        row_stride, column_stride,
                        wrap));
            } else if (row_offset < 0 or column_offset < 0) {
                row_offset = std::max(0, row_offset);
                column_offset = std::max(0, column_offset);
            } else {
                set_arborized_config(
                    new ArborizedConfig(
                        row_field_size, column_field_size,
                        row_stride, column_stride,
                        row_offset, column_offset,
                        wrap));
            }
        }
    }
}

ConnectionConfig::ConnectionConfig(
    bool plastic, int delay, float max_weight,
    ConnectionType type, Opcode opcode)
        : plastic(plastic),
          delay(delay),
          max_weight(max_weight),
          type(type),
          opcode(opcode),
          weight_config(new FlatWeightConfig(1.0)),
          arborized_config(nullptr),
          subset_config(nullptr) { }

ConnectionConfig::ConnectionConfig(
    bool plastic, int delay, float max_weight,
    ConnectionType type, Opcode opcode,
    WeightConfig *weight_config)
        : plastic(plastic),
          delay(delay),
          max_weight(max_weight),
          type(type),
          opcode(opcode),
          weight_config(weight_config),
          arborized_config(nullptr),
          subset_config(nullptr) { }

ConnectionConfig::~ConnectionConfig() {
    delete weight_config;
    if (arborized_config != nullptr) delete arborized_config;
    if (subset_config != nullptr) delete subset_config;
}

bool ConnectionConfig::validate() {
    switch (type) {
        case SUBSET: return subset_config != nullptr;
        case CONVERGENT:
        case CONVOLUTIONAL:
        case DIVERGENT:
            return arborized_config != nullptr;
    }
    return true;
}

std::string ConnectionConfig::str() const {
    std::string str = "[" +
        ConnectionTypeStrings.at(type) + "/" +
        OpcodeStrings.at(opcode) + "/" +
        std::to_string(plastic) + "/" +
        std::to_string(delay) + "/" +
        std::to_string(max_weight) + "/" +
        weight_config->str();

    switch (type) {
        case SUBSET:
            str += ((subset_config == nullptr) ? "" : subset_config->str());
            break;
        case CONVERGENT:
        case CONVOLUTIONAL:
        case DIVERGENT:
            str += ((arborized_config == nullptr) ? "" : arborized_config->str());
            break;
    }

    return str + "]";
}

int ConnectionConfig::get_expected_rows(int rows) {
    switch (type) {
        case ONE_TO_ONE:
            return rows;
        case FULLY_CONNECTED:
            return rows;
        case SUBSET:
            return subset_config->to_row_end - subset_config->to_row_start;
        default:
            int row_field_size = arborized_config->row_field_size;
            int row_stride = arborized_config->row_stride;
            switch(type) {
                case CONVERGENT:
                case CONVOLUTIONAL:
                    if (row_stride == 0) return rows;
                    return std::max(1,
                        1 + ((rows - row_field_size) / row_stride));
                case DIVERGENT:
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
        case ONE_TO_ONE:
            return columns;
        case FULLY_CONNECTED:
            return columns;
        case SUBSET:
            return subset_config->to_col_end - subset_config->to_col_start;
        default:
            int column_field_size = arborized_config->column_field_size;
            int column_stride = arborized_config->column_stride;
            switch(type) {
                case CONVERGENT:
                case CONVOLUTIONAL:
                    if (column_stride == 0) return columns;
                    return std::max(1,
                        1 + ((columns - column_field_size) / column_stride));
                case DIVERGENT:
                    return std::max(1,
                        column_field_size + (column_stride * (columns - 1)));
                default:
                    ErrorManager::get_instance()->log_error(
                        "Invalid call to get_expected_columns!");
            }
    }
}
