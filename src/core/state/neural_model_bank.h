#ifndef neural_model_bank_h
#define neural_model_bank_h

#include <map>
#include <set>

#include "network/layer.h"
#include "network/connection.h"

class Attributes;
class AttributeData;
class WeightMatrix;

/* Typedef for subclass build method
 * This can't be virtual because it's a class method */
typedef Attributes* (*ATT_BUILD_PTR)(Layer *layer);
typedef WeightMatrix* (*MAT_BUILD_PTR)(Connection *conn);

class NeuralModelBank {
    public:
        static NeuralModelBank *get_instance();

        static const std::set<std::string> get_neural_models();
        static OutputType get_output_type(std::string neural_model);

        // Registers an Attributes subclass neural model name with the state
        static bool register_attributes(std::string neural_model,
            OutputType output_type, ATT_BUILD_PTR build_ptr);

        // Registers a WeightMatrix subclass neural model name with the state
        static bool register_weight_matrix(std::string neural_model,
            MAT_BUILD_PTR build_ptr);

        // Builds an instance of an Attributes subclass by name
        static Attributes *build_attributes(Layer *layer);

        // Builds an instance of a WeightMatrix subclass by name
        static WeightMatrix *build_weight_matrix(Connection *conn);

    protected:
        static NeuralModelBank *instance;
        NeuralModelBank() { }

        // Set of neural model implementations
        std::set<std::string> neural_models;
        std::map<std::string, ATT_BUILD_PTR> att_build_pointers;
        std::map<std::string, MAT_BUILD_PTR> mat_build_pointers;
        std::map<std::string, OutputType> output_types;
};

#endif
