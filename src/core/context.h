#ifndef context_h
#define context_h

class Network;
class State;
class Environment;

class Context {
    public:
        Context(Network *network, Environment *env, State *st)
            : network(network), environment(env), state(st) { }

        Network * const network;
        Environment * const environment;
        State * const state;
};

#endif
