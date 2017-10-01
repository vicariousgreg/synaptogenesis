import ctypes

_syn = ctypes.CDLL('synaptogenesis.so')

def create_properties(properties=dict()):
    props = _syn.create_properties()
    for k,v in properties.iteritems():
        if type(v) is dict:
            add_child(props, k, create_properties(v))
        elif type(v) is list:
            for x in v:
                if type(x) is str:
                    add_to_array(props, k, create_properties(x))
        else:
            add_property(props, k, str(v))
    return props

def add_property(properties, key, val):
    global _syn
    _syn.add_property(properties, key, val);

def add_child(properties, key, child):
    global _syn
    _syn.add_child(properties, key, child);

def add_to_array(properties, key, props):
    global _syn
    _syn.add_to_array(properties, key, props);

def create_network(properties):
    global _syn
    return _syn.create_network(create_properties(properties))

def save_network(structure, filename):
    global _syn
    return _syn.save_net(network, filename)

def run(network, iterations, verbose):
    global _syn
    return _syn.run(network, ctypes.c_int(iterations), ctypes.c_bool(verbose))

structure = {"name" : "debug", "type" : "parallel"}

rows = 10
cols = 20

source = {
    "name" : "source",
    "neural model" : "debug",
    "rows" : rows,
    "columns" : cols}
dest = {
    "name" : "dest",
    "neural model" : "debug",
    "rows" : rows,
    "columns" : cols}

structure["layers"] = [source, dest]
connections = [
    {
        "from layer" : "source",
        "to layer" : "dest",
        "type" : "fully connected",
        "opcode" : "add",
        "plastic" : "true",
    },
    {
        "from layer" : "source",
        "to layer" : "dest",
        "type" : "subset",
        "opcode" : "add",
        "plastic" : "true",
        "subset config" : {
            "from row start" : "2",
            "from row end" : "8",
            "from column start" : "2",
            "from column end" : "18",
            "to row start" : "2",
            "to row end" : "8",
            "to column start" : "2",
            "to column end" : "18"}
    },
    {
        "from layer" : "source",
        "to layer" : "dest",
        "type" : "one to one",
        "opcode" : "add",
        "plastic" : "true",
    }
]

network = create_network(
    {"structures" : [structure],
     "connections" : connections})

save_network(network, "test.json")
run(network, 1, True)
