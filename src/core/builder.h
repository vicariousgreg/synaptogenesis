#ifndef builder_h
#define builder_h

#include <string>

class Network;
class Environment;

Network* load_network(std::string path);
void save_network(Network *network, std::string path);

Environment* load_environment(std::string path);
void save_environment(Environment *environment, std::string path);

#endif
