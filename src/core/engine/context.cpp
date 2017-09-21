#include "engine/context.h"
#include "network/network.h"
#include "state/state.h"
#include "io/environment.h"

Context::Context(Network *network, Environment *env, State *st) {
    this->network     = network;
    this->environment = (env == nullptr) ? new Environment()  : env;
    this->state       = (st == nullptr)  ? new State(network) : st;
}

Context::~Context() {
    delete environment;
    delete state;
    delete network;
    for (auto report : reports)
        delete report;
}
