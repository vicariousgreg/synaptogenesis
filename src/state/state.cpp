#include <cstdlib>
#include <cstring>
#include <sstream>

#include "state/state.h"
#include "tools.h"
#include "parallel.h"

State::State(Model *model, Attributes *attributes, int weight_depth)
        : attributes(attributes),
          weight_matrices(new WeightMatrices(model, weight_depth)) { }

State::~State() {
    delete this->weight_matrices;
    delete this->attributes;
}
