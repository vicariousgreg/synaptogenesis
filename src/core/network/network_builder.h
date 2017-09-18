#ifndef network_builder_h
#define network_builder_h

#include "network/network.h"

Network* load_model(std::string path);
void save_model(Network *network, std::string path);

#endif
