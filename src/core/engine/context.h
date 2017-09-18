#ifndef context_h
#define context_h

class Network;
class State;
class Environment;

class Context {
    public:
        Context(Network *network,
            Environment *env = nullptr,
            State *st = nullptr);

        virtual ~Context();

        Network *get_network() { return network; }
        Environment *get_environment() { return environment; }
        State *get_state() { return state; }

    private:
        Network* network;
        Environment* environment;
        State* state;
};

#endif
