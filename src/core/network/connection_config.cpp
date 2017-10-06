#include <sstream>

#include "network/layer.h"
#include "network/connection.h"
#include "network/connection_config.h"
#include "util/error_manager.h"

ArborizedConfig::ArborizedConfig(PropertyConfig *config) {
    row_field_size = config->get_int("row field size", 0);
    column_field_size = config->get_int("column field size", 0);
    row_stride = config->get_int("row stride", 1);
    column_stride = config->get_int("column stride", 1);
    row_offset = config->get_int("row offset", 0);
    column_offset = config->get_int("row offset", 0);
    wrap = config->get_bool("wrap", false);

    if (config->has("field size"))
        row_field_size = column_field_size = config->get_int("field size", 0);
    if (config->has("stride"))
        row_stride = column_stride = config->get_int("stride", 0);
    if (config->has("offset"))
        row_offset = column_offset = config->get_int("offset", 0);

    if (row_field_size <= 0 or column_field_size <= 0)
        LOG_ERROR(
            "Unspecified field size for arborized config!");

    if (row_stride < 0 or column_stride < 0)
        LOG_ERROR(
            "Cannot have negative stride in arborized config!");

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
    if (row_field_size == column_field_size) {
        props.set("field size", std::to_string(row_field_size));
    } else {
        props.set("row field size", std::to_string(row_field_size));
        props.set("column field size", std::to_string(column_field_size));
    }
    if (row_stride == column_stride) {
        props.set("stride", std::to_string(row_stride));
    } else {
        props.set("row stride", std::to_string(row_stride));
        props.set("column stride", std::to_string(column_stride));
    }
    if (row_offset == column_offset) {
        props.set("offset", std::to_string(row_offset));
    } else {
        props.set("row offset", std::to_string(row_offset));
        props.set("column offset", std::to_string(column_offset));
    }
    props.set("wrap", std::to_string(wrap));
    return props;
}

SubsetConfig::SubsetConfig(PropertyConfig *config) {
    from_row_start = config->get_int("from row start", 0);
    from_row_end = config->get_int("from row end", 0);
    to_row_start = config->get_int("to row start", 0);
    to_row_end = config->get_int("to row end", 0);
    from_col_start = config->get_int("from column start", 0);
    from_col_end = config->get_int("from column end", 0);
    to_col_start = config->get_int("to column start", 0);
    to_col_end = config->get_int("to column end", 0);

    from_row_size = from_row_end - from_row_start;
    from_col_size = from_col_end - from_col_start;
    from_size = from_row_size * from_col_size;
    to_row_size = to_row_end - to_row_start;
    to_col_size = to_col_end - to_col_start;
    to_size = to_row_size * to_col_size;
    total_size = from_size * to_size;

    if (from_row_size <= 0 or to_row_size <= 0 or
        from_col_size <= 0 or to_col_size <= 0)
        LOG_ERROR(
            "Invalid subset dimensions!");
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
            LOG_ERROR(
                "SubsetConfig Connected Config cannot have"
                " negative start indices!");
        if (from_row_start > from_row_end or from_col_start > from_col_end
                or to_row_start > to_row_end or to_col_start > to_col_end)
            LOG_ERROR(
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
    props.set("from row start", std::to_string(from_row_start));
    props.set("from row end", std::to_string(from_row_end));
    props.set("from column start", std::to_string(from_col_start));
    props.set("from column end", std::to_string(from_col_end));
    props.set("to row start", std::to_string(to_row_start));
    props.set("to row end", std::to_string(to_row_end));
    props.set("to column start", std::to_string(to_col_start));
    props.set("to column end", std::to_string(to_col_end));
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

ConnectionConfig::ConnectionConfig(const PropertyConfig *config)
        : PropertyConfig(config),
          name(config->get("name", "")),
          from_layer(config->get("from layer", "")),
          to_layer(config->get("to layer", "")),
          dendrite(config->get("dendrite", "root")),
          plastic(config->get_bool("plastic", true)),
          delay(config->get_int("delay", 0)),
          max_weight(config->get_float("max weight", 1.0)),
          type(get_connection_type(config->get("type", "fully connected"))),
          opcode(get_opcode(config->get("opcode", "add"))) {
    if (not config->has("from layer"))
        LOG_ERROR(
            "Attempted to construct ConnectionConfig "
            "without source layer!");
    if (not config->has("to layer"))
        LOG_ERROR(
            "Attempted to construct ConnectionConfig "
            "without destination layer!");
    if (delay < 0)
        LOG_ERROR(
            "Attempted to construct ConnectionConfig with negative delay!");
    switch (type) {
        case SUBSET:
            if (not has_child("subset config"))
                LOG_ERROR(
                    "Attempted to create SUBSET connection without "
                    "specifying subset configuration!");
            break;
        case CONVERGENT:
        case CONVOLUTIONAL:
        case DIVERGENT:
            if (not has_child("arborized config"))
                LOG_ERROR(
                    "Attempted to create arborized connection without "
                    "specifying subset configuration!");
            break;
    }
}

ConnectionConfig::ConnectionConfig(
    std::string from_layer, std::string to_layer,
    bool plastic, int delay, float max_weight,
    ConnectionType type, Opcode opcode,
    PropertyConfig *specialized_config,
    PropertyConfig *weight_config,
    std::string dendrite, std::string name)
        : name(name),
          from_layer(from_layer),
          to_layer(to_layer),
          dendrite(dendrite),
          plastic(plastic),
          delay(delay),
          max_weight(max_weight),
          type(type),
          opcode(opcode) {
    this->set("name", name);
    this->set("to layer", to_layer);
    this->set("from layer", from_layer);
    this->set("dendrite", dendrite);
    this->set("plastic", (plastic) ? "true" : "false");
    this->set("delay", std::to_string(delay));
    this->set("max weight", std::to_string(max_weight));
    this->set("type", ConnectionTypeStrings.at(type));
    this->set("opcode", OpcodeStrings.at(opcode));

    if (delay < 0)
        LOG_ERROR(
            "Attempted to construct ConnectionConfig with negative delay!");

    if (weight_config != nullptr)
        this->set_child("weight config", weight_config);

    switch (type) {
        case SUBSET:
            if (specialized_config == nullptr)
                LOG_ERROR(
                    "Attempted to create SUBSET connection without "
                    "specifying subset configuration!");
            this->set_child("subset config", specialized_config);
            break;
        case CONVERGENT:
        case CONVOLUTIONAL:
        case DIVERGENT:
            if (specialized_config == nullptr)
                LOG_ERROR(
                    "Attempted to create arborized connection without "
                    "specifying subset configuration!");
            this->set_child("arborized config", specialized_config);
            break;
    }
}

bool ConnectionConfig::validate(Connection *conn) const {
    if (type == SUBSET)
        return get_subset_config().validate(conn);
    return true;
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

int ConnectionConfig::get_expected_rows(int rows) const {
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
                    LOG_ERROR(
                        "Invalid call to get_expected_rows!");
            }
    }
}

int ConnectionConfig::get_expected_columns(int columns) const {
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
                    LOG_ERROR(
                        "Invalid call to get_expected_columns!");
            }
    }
}
