#ifndef context_h
#define context_h

class Model;
class State;
class EnvironmentModel;

class Context {
    public:
        Context(Model *model,
            EnvironmentModel *env = nullptr,
            State *st = nullptr);

        virtual ~Context();

        Model *get_model() { return model; }
        EnvironmentModel *get_environment_model() { return environment_model; }
        State *get_state() { return state; }

    private:
        Model* model;
        EnvironmentModel* environment_model;
        State* state;
};

#endif
