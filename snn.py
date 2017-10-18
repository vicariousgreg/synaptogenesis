from syngen import Network, Environment, create_callback, FloatArray

# Create main structure (feedforward engine)
structure = {"name" : "snn", "type" : "parallel"}

dim = 64
input_grid = 4
exc_exc_spread = 31
exc_exc_self_spread = 19
inh_inh_self_spread = 7
exc_inh_spread = 21
inh_exc_spread = 11

# Excitatory layers
layer = {
    "neural model" : "leaky_izhikevich",
    "rows" : dim,
    "columns" : dim,
    "noise config" : {
        "type" : "poisson",
        "value" : 0.1,
        "rate" : 10
    },
    "neuron spacing" : "0.1",
    "init" : "random positive"
}

layer_a = dict(layer)
layer_a["name"] = "layer_a"

association_layer = dict(layer)
association_layer["name"] = "association_layer"
association_layer["rows"] = dim*2
association_layer["columns"] = dim*2
association_layer["neuron spacing"] = "0.05"

layer_b = dict(layer)
layer_b["name"] = "layer_b"

# Input and gate layers
input_layer = {
    "neural model" : "relay",
    "rows" : dim,
    "columns" : dim,
    "noise config" : {
        "type" : "poisson",
        "value" : 1.0,
        "rate" : 10
    },
    "neuron spacing" : "0.1",
    "init" : "random positive"
}
gate_layer = {
    "neural model" : "relay",
    "rows" : input_grid,
    "columns" : input_grid
}

input_a = dict(input_layer)
input_a["name"] = "input_a"

input_b = dict(input_layer)
input_b["name"] = "input_b"

gate_a = dict(gate_layer)
gate_a["name"] = "gate_a"

gate_b = dict(gate_layer)
gate_b["name"] = "gate_b"

# Inhibitory layers
layer_inh = {
    "neural model" : "izhikevich",
    "rows" : dim/2,
    "columns" : dim/2,
    "noise config" : {
        "type" : "poisson",
        "value" : 0.1,
        "rate" : 10
    },
    "neuron spacing" : "0.2",
    "init" : "random negative"
}

layer_a_inh = dict(layer_inh)
layer_a_inh["name"] = "layer_a_inh"

association_layer_inh = dict(layer_inh)
association_layer_inh["name"] = "association_layer_inh"
association_layer_inh["rows"] = dim
association_layer_inh["columns"] = dim
association_layer_inh["neuron spacing"] = "0.1"

layer_b_inh = dict(layer_inh)
layer_b_inh["name"] = "layer_b_inh"

# Add layers to structure
structure["layers"] = [
    layer_a, association_layer, layer_b,
    layer_a_inh, association_layer_inh, layer_b_inh,
    input_a, input_b, gate_a, gate_b]

exc_exc = {
    "type" : "convergent",
    "arborized config" : {
        "field size" : exc_exc_spread,
        "stride" : "2",
        "wrap" : "true",
    },
    "opcode" : "add",
    "plastic" : "true",
#    "learning rate" : "0.01",
    "max" : "0.5",
    "weight config" : {
        "type" : "power law",
        "exponent" : "1.5",
#        "type" : "flat",
#        "weight" : "0.1",
        "fraction" : "0.1"
    },
    "myelinated" : "false"
}
exc_exc_self = {
    "type" : "convergent",
    "arborized config" : {
        "field size" : exc_exc_self_spread,
        "wrap" : "true",
    },
    "opcode" : "add",
    "plastic" : "true",
#    "learning rate" : "0.01",
    "max" : "0.5",
    "weight config" : {
        "type" : "power law",
        "exponent" : "1.5",
#        "type" : "flat",
#        "weight" : "0.1",
        "fraction" : "0.1"
    },
}

a_ass = dict(exc_exc)
a_ass["from layer"] = "layer_a"
a_ass["to layer"] = "association_layer"
a_ass["type"] = "divergent"
ass_b = dict(exc_exc)
ass_b["from layer"] = "association_layer"
ass_b["to layer"] = "layer_b"
b_ass = dict(exc_exc)
b_ass["from layer"] = "layer_b"
b_ass["to layer"] = "association_layer"
b_ass["type"] = "divergent"
ass_a = dict(exc_exc)
ass_a["from layer"] = "association_layer"
ass_a["to layer"] = "layer_a"

exc_inh = {
    "type" : "convergent",
    "arborized config" : {
        "field size" : exc_inh_spread,
        "stride" : 2,
        "wrap" : "true",
    },
    "opcode" : "add",
    "plastic" : "true",
    "max" : "0.5",
    "weight config" : {
        "type" : "flat",
        "weight" : "0.2",
        "fraction" : "0.1"
    },
}
inh_exc = {
    "type" : "divergent",
    "arborized config" : {
        "field size" : inh_exc_spread,
        "stride" : 2,
        "wrap" : "true",
    },
    "opcode" : "sub",
    "plastic" : "false",
    "weight config" : {
        "type" : "flat",
        "weight" : "1.0",
        "fraction" : "0.3"
    },
    "myelinated" : "true"
}
inh_inh_self = {
    "type" : "convergent",
    "arborized config" : {
        "field size" : inh_inh_self_spread,
        "wrap" : "true",
    },
    "opcode" : "gap",
    "plastic" : "false",
    "max" : "0.5",
    "weight config" : {
        "type" : "flat",
        "weight" : "0.01",
        "fraction" : "0.1"
    },
}

a_a = dict(exc_inh)
a_a["from layer"] = "layer_a"
a_a["to layer"] = "layer_a_inh"

ass_ass = dict(exc_inh)
ass_ass["from layer"] = "association_layer"
ass_ass["to layer"] = "association_layer_inh"

b_b = dict(exc_inh)
b_b["from layer"] = "layer_b"
b_b["to layer"] = "layer_b_inh"

a_a_inh = dict(inh_exc)
a_a_inh["from layer"] = "layer_a_inh"
a_a_inh["to layer"] = "layer_a"

ass_ass_inh = dict(inh_exc)
ass_ass_inh["from layer"] = "association_layer_inh"
ass_ass_inh["to layer"] = "association_layer"

b_b_inh = dict(inh_exc)
b_b_inh["from layer"] = "layer_b_inh"
b_b_inh["to layer"] = "layer_b"

a_self = dict(exc_exc_self)
a_self["from layer"] = "layer_a"
a_self["to layer"] = "layer_a"

ass_self = dict(exc_exc_self)
ass_self["from layer"] = "association_layer"
ass_self["to layer"] = "association_layer"

b_self = dict(exc_exc_self)
b_self["from layer"] = "layer_b"
b_self["to layer"] = "layer_b"

a_self_inh = dict(inh_inh_self)
a_self_inh["from layer"] = "layer_a_inh"
a_self_inh["to layer"] = "layer_a_inh"

ass_self_inh = dict(inh_inh_self)
ass_self_inh["from layer"] = "association_layer_inh"
ass_self_inh["to layer"] = "association_layer_inh"

b_self_inh = dict(inh_inh_self)
b_self_inh["from layer"] = "layer_b_inh"
b_self_inh["to layer"] = "layer_b_inh"


input_conn = {
    "type" : "one to one",
    "opcode" : "add",
    "plastic" : "false",
    "max" : "1.0",
    "weight config" : {
        "type" : "flat",
        "weight" : "1.0",
        "fraction" : "1.0"
    },
    "myelinated" : "true"
}
gate_conn = {
    "type" : "divergent",
    "opcode" : "mult",
    "plastic" : "false",
    "max" : "1.0",
    "weight config" : {
        "type" : "flat",
        "weight" : "1.0",
        "fraction" : "1.0"
    },
    "arborized config" : {
        "field size" : dim/input_grid,
        "stride" : dim/input_grid,
        "wrap" : "false",
        "offset" : "0"
    },
    "myelinated" : "true"
}


in_a = dict(input_conn)
in_a["from layer"] = "input_a"
in_a["to layer"] = "layer_a"
in_a["short term plasticity"] = "false"
g_a = dict(gate_conn)
g_a["from layer"] = "gate_a"
g_a["to layer"] = "input_a"
g_a["short term plasticity"] = "false"

in_b = dict(input_conn)
in_b["from layer"] = "input_b"
in_b["to layer"] = "layer_b"
in_b["short term plasticity"] = "false"
g_b = dict(gate_conn)
g_b["from layer"] = "gate_b"
g_b["to layer"] = "input_b"
g_b["short term plasticity"] = "false"



# Create connections
connections = [
    in_a, g_a,
    in_b, g_b,
    a_ass, ass_a,
    b_ass, ass_b,
    a_self, ass_self, b_self,
    a_a, ass_ass, b_b,
    a_a_inh, ass_ass_inh, b_b_inh,
    a_self_inh, ass_self_inh, b_self_inh,
]

change_rate = 10000
modules = [
    {
        "type" : "visualizer",
        "layers" : [
            { "structure" : "snn", "layer" : "layer_a" },
            { "structure" : "snn", "layer" : "association_layer" },
            { "structure" : "snn", "layer" : "layer_b" },
            { "structure" : "snn", "layer" : "layer_a_inh" },
            { "structure" : "snn", "layer" : "association_layer_inh" },
            { "structure" : "snn", "layer" : "layer_b_inh" },
            { "structure" : "snn", "layer" : "input_a" },
            { "structure" : "snn", "layer" : "input_b" },
            { "structure" : "snn", "layer" : "gate_a" },
            { "structure" : "snn", "layer" : "gate_b" },
        ]
    },
    {
        "type" : "heatmap",
        "rate" : 1000000, # Long term
        "linear" : "true",
        "layers" : [
            { "structure" : "snn", "layer" : "layer_a" },
            { "structure" : "snn", "layer" : "association_layer" },
            { "structure" : "snn", "layer" : "layer_b" },
            { "structure" : "snn", "layer" : "layer_a_inh" },
            { "structure" : "snn", "layer" : "association_layer_inh" },
            { "structure" : "snn", "layer" : "layer_b_inh" },
            { "structure" : "snn", "layer" : "input_a" },
            { "structure" : "snn", "layer" : "input_b" },
            { "structure" : "snn", "layer" : "gate_a" },
            { "structure" : "snn", "layer" : "gate_b" },
        ]
    },
    {
        "type" : "heatmap",
        "rate" : change_rate, # Short term
        "linear" : "false",
        "layers" : [
            { "structure" : "snn", "layer" : "layer_a" },
            { "structure" : "snn", "layer" : "association_layer" },
            { "structure" : "snn", "layer" : "layer_b" },
            { "structure" : "snn", "layer" : "layer_a_inh" },
            { "structure" : "snn", "layer" : "association_layer_inh" },
            { "structure" : "snn", "layer" : "layer_b_inh" },
            { "structure" : "snn", "layer" : "input_a" },
            { "structure" : "snn", "layer" : "input_b" },
            { "structure" : "snn", "layer" : "gate_a" },
            { "structure" : "snn", "layer" : "gate_b" },
        ]
    },
    {
        "type" : "one_hot_random_input",
        "random" : "false",
        "val" : "1.0",
        "rate" : change_rate,
        "layers" : [
            { "structure" : "snn", "layer" : "gate_a" },
        ]
    },
    {
        "type" : "one_hot_random_input",
        "random" : "false",
        "val" : "1.0",
        "rate" : change_rate,
        "layers" : [
            { "structure" : "snn", "layer" : "gate_b" },
        ]
    },
]

env = Environment({"modules" : modules})

# Create network
network = Network(
    {"structures" : [structure],
     "connections" : connections})

print(network.run(env, {"multithreaded" : "true",
                        "worker threads" : "1",
                        "verbose" : "true"}))

# Delete the objects
del network
del env
