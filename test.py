import ctypes

_syn = ctypes.CDLL('synaptogenesis.so')

def create_network():
    global _syn
    return _syn.create_network()

def print_network(network):
    global _syn
    _syn.print_network(network)

network = create_network()
print_network(network);
