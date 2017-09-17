#include "context.h"
#include "model/model.h"
#include "state/state.h"
#include "io/environment_model.h"

Context::Context(Model *model,
    EnvironmentModel *env,
    State *st)
    : model(model),
      environment_model(env==nullptr ? new EnvironmentModel() : env),
      state(st==nullptr ? new State(model) : st) { }

Context::~Context() {
    delete environment_model;
    delete state;
    delete model;
}
