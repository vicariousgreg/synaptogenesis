#include <sstream>

#include "network/layer.h"
#include "network/connection.h"
#include "network/connection_config.h"
#include "util/error_manager.h"

ArborizedConfig::ArborizedConfig(PropertyConfig *config) {
    row_field_size = -1;
    column_field_size = -1;
    row_stride = 1;
    column_stride = 1;
    row_offset = -1;
    column_offset = -1;
    wrap = false;

    for (auto pair : config->get()) {
        if (pair.first == "row field size")
            row_field_size = std::stoi(pair.second);
        else if (pair.first == "column field size")
            column_field_size = std::stoi(pair.second);
        else if (pair.first == "field size")
            row_field_size = column_field_size =
                std::stoi(pair.second);
        else if (pair.first == "row stride")
            row_stride = std::stoi(pair.second);
        else if (pair.first == "column stride")
            column_stride = std::stoi(pair.second);
        else if (pair.first == "stride")
            row_stride = column_stride = std::stoi(pair.second);
        else if (pair.first == "row offset")
            row_offset = std::stoi(pair.second);
        else if (pair.first == "column offset")
            column_offset = std::stoi(pair.second);
        else if (pair.first == "offset")
            row_offset = column_offset = std::stoi(pair.second);
        else if (pair.first == "wrap")
            wrap = pair.second == "true";
        else
            ErrorManager::get_instance()->log_error(
                "Unrecognized arborized config property: " + pair.first);
    }

    if (row_field_size < 0 or column_field_size < 0)
        ErrorManager::get_instance()->log_error(
            "Unspecified field size for arborized config!");

    // If offsets are not provided, use default
    if (not config->has("row offset") and not config->has("offset"))
        row_offset = -row_field_size/2;
    if (not config->has("column offset") and not config->has("offset"))
        column_offset = -row_field_size/2;
}

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

PropertyConfig ArborizedConfig::to_property_config() const {
    PropertyConfig props;
    props.set_value("row field size", std::to_string(row_field_size));
    props.set_value("column field size", std::to_string(column_field_size));
    props.set_value("row stride", std::to_string(row_stride));
    props.set_value("column stride", std::to_string(column_stride));
    props.set_value("row offset", std::to_string(row_offset));
    props.set_value("column offset", std::to_string(column_offset));
    props.set_value("wrap", std::to_string(wrap));
    return props;
}

SubsetConfig::SubsetConfig(PropertyConfig *config) {
    from_row_start = from_row_end =
        from_col_start = from_col_end =
        to_row_start = to_row_end =
        to_col_start = to_col_end = 0;

    for (auto pair : config->get()) {
        if (pair.first == "from row start")
            from_row_start = std::stoi(pair.second);
        else if (pair.first == "from row end")
            from_row_end = std::stoi(pair.second);
        else if (pair.first == "from column start")
            from_col_start = std::stoi(pair.second);
        else if (pair.first == "from column end")
            from_col_end = std::stoi(pair.second);
        else if (pair.first == "to row start")
            to_row_start = std::stoi(pair.second);
        else if (pair.first == "to row end")
            to_row_end = std::stoi(pair.second);
        else if (pair.first == "to column start")
            to_col_start = std::stoi(pair.second);
        else if (pair.first == "to column end")
            to_col_end = std::stoi(pair.second);
        else
            ErrorManager::get_instance()->log_error(
                "Unrecognized subset config property: " + pair.first);
    }

    from_row_size = from_row_end - from_row_start;
    from_col_size = from_col_end - from_col_start;
    from_size = from_row_size * from_col_size;
    to_row_size = to_row_end - to_row_start;
    to_col_size = to_col_end - to_col_start;
    to_size = to_row_size * to_col_size;
    total_size = from_size * to_size;
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

bool SubsetConfig::validate(Connection *conn) const {
    Layer *from_layer = conn->from_layer;
    Layer *to_layer = conn->to_layer;
    return
        from_row_end <= from_layer->rows
        and from_col_end <= from_layer->columns
        and to_row_end <= to_layer->rows
        and to_col_end <= to_layer->columns;
}

PropertyConfig SubsetConfig::to_property_config() const {
    PropertyConfig props;
    props.set_value("from row start", std::to_string(from_row_start));
    props.set_value("from row end", std::to_string(from_row_end));
    props.set_value("from column start", std::to_string(from_col_start));
    props.set_value("from column end", std::to_string(from_col_end));
    props.set_value("to row start", std::to_string(to_row_start));
    props.set_value("to row end", std::to_string(to_row_end));
    props.set_value("to column start", std::to_string(to_col_start));
    props.set_value("to column end", std::to_string(to_col_end));
    return props;
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
    : PropertyConfig(config),
      plastic(config->get("plastic", "true") == "true"),
      delay(std::stoi(config->get("delay", "0"))),
      max_weight(std::stof(config->get("max", "1.0"))),
      type(ConnectionTypes.at(config->get("type", "fully connected"))),
      opcode(Opcodes.at(config->get("opcode", "add"))) { }

ConnectionConfig::ConnectionConfig(
    bool plastic, int delay, float max_weight,
    ConnectionType type, Opcode opcode)
        : plastic(plastic),
          delay(delay),
          max_weight(max_weight),
          type(type),
          opcode(opcode) { }

ConnectionConfig::ConnectionConfig(
    bool plastic, int delay, float max_weight,
    ConnectionType type, Opcode opcode,
    PropertyConfig *weight_config)
        : plastic(plastic),
          delay(delay),
          max_weight(max_weight),
          type(type),
          opcode(opcode) {
    this->set_child("weight config", weight_config);
}

ConnectionConfig::~ConnectionConfig() { }

bool ConnectionConfig::validate(Connection *conn) const {
    if (type == SUBSET)
        return get_subset_config().validate(conn);
    return true;
}

ConnectionConfig* ConnectionConfig::set_arborized_config(
        ArborizedConfig *config) {
    auto props = config->to_property_config();
    return this->set_arborized_config(&props);
}

ConnectionConfig* ConnectionConfig::set_subset_config(SubsetConfig *config) {
    auto props = config->to_property_config();
    return this->set_subset_config(&props);
}

ConnectionConfig* ConnectionConfig::set_arborized_config(
        PropertyConfig *config) {
    if (not this->has_child("arborized config")) {
        this->set_child("arborized config", new PropertyConfig(config));
    } else {
        auto child = this->get_child("arborized config");
        for (auto pair : config->get())
            child->set_value(pair.first, pair.second);
    }
    return this;
}

ConnectionConfig* ConnectionConfig::set_subset_config(PropertyConfig *config) {
    if (not this->has_child("subset config")) {
        this->set_child("subset config", new PropertyConfig(config));
    } else {
        auto child = this->get_child("subset config");
        for (auto pair : config->get())
            child->set_value(pair.first, pair.second);
    }
    return this;
}

ConnectionConfig* ConnectionConfig::set_weight_config(PropertyConfig *config) {
    if (not this->has_child("weight config")) {
        this->set_child("weight config", new PropertyConfig(config));
    } else {
        auto child = this->get_child("weight config");
        for (auto pair :config->get())
            child->set_value(pair.first, pair.second);
    }
    return this;
}

/* Specialized config getters */
const ArborizedConfig ConnectionConfig::get_arborized_config() const {
    if (this->has_child("arborized config"))
        return ArborizedConfig(this->get_child("arborized config"));
    else
        return ArborizedConfig();
}

const SubsetConfig ConnectionConfig::get_subset_config() const {
    if (this->has_child("subset config"))
        return SubsetConfig(this->get_child("subset config"));
    else
        return SubsetConfig();
}

const PropertyConfig ConnectionConfig::get_weight_config() const {
    if (this->has_child("weight config"))
        return PropertyConfig(this->get_child("weight config"));
    else
        return PropertyConfig();
}

std::string ConnectionConfig::str() const {
    std::string str = "[" +
        ConnectionTypeStrings.at(type) + "/" +
        OpcodeStrings.at(opcode) + "/" +
        std::to_string(plastic) + "/" +
        std::to_string(delay) + "/" +
        std::to_string(max_weight) + "/" +
        get_weight_config().str();

    switch (type) {
        case SUBSET:
            str += get_subset_config().str();
            break;
        case CONVERGENT:
        case CONVOLUTIONAL:
        case DIVERGENT:
            str += get_arborized_config().str();
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
        case SUBSET: {
            auto subset_config = get_subset_config();
            return subset_config.to_row_end - subset_config.to_row_start;
        }
        default:
            auto arborized_config = get_arborized_config();
            int row_field_size = arborized_config.row_field_size;
            int row_stride = arborized_config.row_stride;
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
        case SUBSET: {
            auto subset_config = get_subset_config();
            return subset_config.to_col_end - subset_config.to_col_start;
        }
        default:
            auto arborized_config = get_arborized_config();
            int column_field_size = arborized_config.column_field_size;
            int column_stride = arborized_config.column_stride;
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
