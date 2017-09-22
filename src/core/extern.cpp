#include "extern.h"
#include "network/network.h"
#include "network/structure.h"

Net create_network() {
    printf("Creating network...\n");
    auto network = new Network();
    network->add_structure(new Structure("test"));
    return network;
}

void print_network(Net network) {
    printf("Name: %s\n", ((Network*)network)->get_structures()[0]->name.c_str());
}
