import ctypes

_syn = ctypes.CDLL('synaptogenesis.so')

def create_properties(properties=dict(), children=dict()):
    props = _syn.create_properties()
    for k,v in properties.iteritems():
        add_property(props, k, v)
    for k,v in children.iteritems():
        add_child(props, k, create_properties(v))
    return props

def add_property(properties, key, val):
    global _syn
    _syn.add_property(properties, key, val);

def add_child(properties, key, child):
    global _syn
    _syn.add_child(properties, key, child);

def create_network():
    global _syn
    return _syn.create_network()

def create_structure(name, cluster_type):
    global _syn
    return _syn.create_structure(name, cluster_type)

def add_structure(network, structure):
    global _syn
    return _syn.add_structure(network, structure)

def add_layer(structure, properties):
    global _syn
    return _syn.add_layer(structure, properties)

def add_dendrite(structure, layer, parent, child):
    global _syn
    return _syn.add_dendrite(structure, layer, parent, child)

def connect_layers(from_structure, from_layer, to_structure, to_layer, props):
    global _syn
    return _syn.connect_layers(
        from_structure, from_layer,
        to_structure, to_layer, props)

def save_network(structure, filename):
    global _syn
    return _syn.save_net(network, filename)

def run(network, iterations, verbose):
    global _syn
    return _syn.run(network, ctypes.c_int(iterations), ctypes.c_bool(verbose))

network = create_network()
structure = create_structure("debug", "parallel")
add_structure(network, structure)

add_layer(structure,
    create_properties(
        dict([
            ("name", "source"),
            ("neural model", "debug"),
            ("rows", "10"),
            ("columns", "20"),
        ])))

add_layer(structure,
    create_properties(
        dict([
            ("name", "dest"),
            ("neural model", "debug"),
            ("rows", "10"),
            ("columns", "20"),
        ])))

connect_layers(structure, "source", structure, "dest",
    create_properties(
        dict([
            ("type", "fully connected"),
            ("opcode", "add"),
            ("plastic", "true"),
            ("delay", "0"),
            ("max", "1"),
        ])))

connect_layers(structure, "source", structure, "dest",
    create_properties(
        dict([
            ("type", "subset"),
            ("opcode", "add"),
            ("plastic", "true"),
            ("delay", "0"),
            ("max", "1"),
        ]),
        dict([
            ("subset config", dict([
                ("from row start", "2"),
                ("from row end", "8"),
                ("from column start", "2"),
                ("from column end", "18"),

                ("to row start", "2"),
                ("to row end", "8"),
                ("to column start", "2"),
                ("to column end", "18"),
            ]))
        ])))

connect_layers(structure, "source", structure, "dest",
    create_properties(
        dict([
            ("type", "one to one"),
            ("opcode", "add"),
            ("plastic", "true"),
            ("delay", "0"),
            ("max", "1"),
        ])))

save_network(network, "test.json")
run(network, 1, True)
