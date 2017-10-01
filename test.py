import ctypes

_syn = ctypes.CDLL('synaptogenesis.so')

class CObject:
    def __init(self):
        self.obj = None
    def __del__(self):
        global _syn
        if _syn is not None and self.obj is not None:
            _syn.destroy(self.obj)

class Properties(CObject):
    def __init__(self, props=dict()):
        self.obj = _syn.create_properties()

        # Dictionaries of Property Objects
        self.properties = dict()
        self.children = dict()
        self.arrays = dict()

        for k,v in props.iteritems():
            if type(v) is dict:
                self.add_child(k, v)
            elif type(v) is list:
                for x in v:
                    if type(x) is dict:
                        self.add_to_array(k, x)
            else:
                self.add_property(k, v)

    def add_property(self, key, val):
        global _syn
        self.properties[key] = val
        _syn.add_property(self.obj, key, str(val));

    def add_child(self, key, child):
        global _syn
        child = Properties(child)
        self.children[key] = child
        _syn.add_child(self.obj, key, child.obj);

    def add_to_array(self, key, dictionary):
        global _syn
        props = Properties(dictionary)
        if key not in self.arrays:
            self.arrays[key] = []
        self.arrays[key].append(props)
        _syn.add_to_array(self.obj, key, props.obj);

class Network(CObject):
    def __init__(self, dictionary):
        global _syn
        self.properties = Properties(dictionary)
        self.obj = _syn.create_network(self.properties.obj)

    def save(self, filename):
        global _syn
        return _syn.save_net(self.obj, filename)

    def run(self, iterations, verbose):
        global _syn
        return _syn.run(self.obj, ctypes.c_int(iterations), ctypes.c_bool(verbose))

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

network = Network(
    {"structures" : [structure],
     "connections" : connections})

network.save("test.json")
network.run(1, True)

del network
