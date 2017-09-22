#include "engine/context.h"
#include "network/network.h"
#include "state/state.h"
#include "io/environment.h"

Context::Context(Network *network, Environment *env, State *st)
        : network(network), environment(env), state(st) { }

Context::~Context() {
    delete environment;
    delete state;
    delete network;
    for (auto report : reports)
        delete report;
}

void Context::set_network(Network *net) {
    delete state;
    this->state = nullptr;
    this->network = net;
}

void Context::set_environment(Environment *env) {
    if (this->environment != nullptr)
        delete this->environment;
    this->environment = env;
}

void Context::set_state(State *st) {
    if (this->state != nullptr)
        delete this->state;
    this->state = state;
}

Network* Context::get_network() { return network; }
Environment* Context::get_environment() {
    if (environment == nullptr)
        environment = new Environment();
    return environment;
}
State* Context::get_state() {
    if (state == nullptr)
        state = new State(network);
    return state;
}
