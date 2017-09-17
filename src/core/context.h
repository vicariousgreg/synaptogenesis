#ifndef context_h
#define context_h

#include "model/model.h"
#include "state/state.h"
#include "io/environment.h"
#include "engine/engine.h"

class Context {
    public:
        Context(Model *model,
            State *st=nullptr,
            Environment *env=nullptr,
            Engine *eng=nullptr)
            : model(model),
              state(st==nullptr ? new State(model) : st),
              environment(env==nullptr ? new Environment(state) : env),
              engine(eng==nullptr ? new Engine(state, environment) : eng) { }

        virtual ~Context() {
            delete engine;
            delete environment;
            delete state;
            delete model;
        }

        Model * const model;
        State * const state;
        Environment * const environment;
        Engine * const engine;
};

#endif
