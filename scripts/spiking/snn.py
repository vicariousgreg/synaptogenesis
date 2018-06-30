from syngen import Network, Environment, create_io_callback, FloatArray
from syngen import get_gpus
from syngen import set_suppress_output, set_warnings, set_debug
from copy import deepcopy

set_warnings(False)

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
    "neural model" : "izhikevich",
    "rows" : dim,
    "columns" : dim,
    "init config" : {
        "type" : "poisson",
        "value" : 0.1,
        "rate" : 10
    },
    "neuron spacing" : "0.1",
    "params" : "random positive"
}

layer_a = deepcopy(layer)
layer_a["name"] = "layer_a"

association_layer = deepcopy(layer)
association_layer["name"] = "association_layer"
association_layer["rows"] = dim*2
association_layer["columns"] = dim*2
association_layer["neuron spacing"] = "0.05"

layer_b = deepcopy(layer)
layer_b["name"] = "layer_b"

# Input and gate layers
input_layer = {
    "neural model" : "relay",
    "rows" : dim,
    "columns" : dim,
    "init config" : {
        "type" : "poisson",
        "value" : 1.0,
        "rate" : 10
    },
    "neuron spacing" : "0.1",
    "params" : "random positive"
}
gate_layer = {
    "neural model" : "relay",
    "rows" : input_grid,
    "columns" : input_grid
}

input_a = deepcopy(input_layer)
input_a["name"] = "input_a"

input_b = deepcopy(input_layer)
input_b["name"] = "input_b"

gate_a = deepcopy(gate_layer)
gate_a["name"] = "gate_a"

gate_b = deepcopy(gate_layer)
gate_b["name"] = "gate_b"

# Inhibitory layers
layer_inh = {
    "neural model" : "izhikevich",
    "rows" : dim/2,
    "columns" : dim/2,
    "init config" : {
        "type" : "poisson",
        "value" : 0.1,
        "rate" : 10
    },
    "neuron spacing" : "0.2",
    "params" : "random negative"
}

layer_a_inh = deepcopy(layer_inh)
layer_a_inh["name"] = "layer_a_inh"

association_layer_inh = deepcopy(layer_inh)
association_layer_inh["name"] = "association_layer_inh"
association_layer_inh["rows"] = dim
association_layer_inh["columns"] = dim
association_layer_inh["neuron spacing"] = "0.1"

layer_b_inh = deepcopy(layer_inh)
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
        "wrap" : True,
    },
    "opcode" : "add",
    "plastic" : True,
    "learning rate" : "0.001",
    "max" : "0.5",
    "weight config" : {
#        "type" : "power law",
#        "exponent" : "1.5",
        "type" : "flat",
        "weight" : "0.01",
        "fraction" : "0.1",
        "circular mask" : { }
    },
    "myelinated" : False
}
exc_exc_self = {
    "type" : "convergent",
    "arborized config" : {
        "field size" : exc_exc_self_spread,
        "wrap" : True,
    },
    "opcode" : "add",
    "plastic" : True,
    "learning rate" : "0.001",
    "max" : "0.5",
    "weight config" : {
#        "type" : "power law",
#        "exponent" : "1.5",
        "type" : "flat",
        "weight" : "0.01",
        "fraction" : "0.1",
        "diagonal" : False,
        "circular mask" : { }
    },
}

a_ass = deepcopy(exc_exc)
a_ass["from layer"] = "layer_a"
a_ass["to layer"] = "association_layer"
a_ass["type"] = "divergent"
del a_ass["weight config"]["circular mask"]
ass_b = deepcopy(exc_exc)
ass_b["from layer"] = "association_layer"
ass_b["to layer"] = "layer_b"
b_ass = deepcopy(exc_exc)
b_ass["from layer"] = "layer_b"
b_ass["to layer"] = "association_layer"
b_ass["type"] = "divergent"
del b_ass["weight config"]["circular mask"]
ass_a = deepcopy(exc_exc)
ass_a["from layer"] = "association_layer"
ass_a["to layer"] = "layer_a"

exc_inh = {
    "type" : "convergent",
    "arborized config" : {
        "field size" : exc_inh_spread,
        "stride" : 2,
        "wrap" : True,
    },
    "opcode" : "add",
    "plastic" : True,
    "learning rate" : "0.001",
    "max" : "0.5",
    "weight config" : {
        "type" : "flat",
        "weight" : "0.01",
        "fraction" : "0.1",
        "circular mask" : { }
    },
}
inh_exc = {
    "type" : "divergent",
    "arborized config" : {
        "field size" : inh_exc_spread,
        "stride" : 2,
        "wrap" : True,
    },
    "opcode" : "sub",
    "plastic" : True,
    "learning rate" : "0.001",
    "weight config" : {
        "type" : "flat",
        "weight" : "0.02",
        "fraction" : "0.3"
    },
    "myelinated" : True
}
inh_inh_self = {
    "type" : "convergent",
    "arborized config" : {
        "field size" : inh_inh_self_spread,
        "wrap" : True,
    },
    "opcode" : "gap",
    "plastic" : False,
    "max" : "0.5",
    "weight config" : {
        "type" : "flat",
        "weight" : "0.01",
        "fraction" : "0.1",
        "circular mask" : { }
    },
}

a_a = deepcopy(exc_inh)
a_a["from layer"] = "layer_a"
a_a["to layer"] = "layer_a_inh"

ass_ass = deepcopy(exc_inh)
ass_ass["from layer"] = "association_layer"
ass_ass["to layer"] = "association_layer_inh"

b_b = deepcopy(exc_inh)
b_b["from layer"] = "layer_b"
b_b["to layer"] = "layer_b_inh"

a_a_inh = deepcopy(inh_exc)
a_a_inh["from layer"] = "layer_a_inh"
a_a_inh["to layer"] = "layer_a"

ass_ass_inh = deepcopy(inh_exc)
ass_ass_inh["from layer"] = "association_layer_inh"
ass_ass_inh["to layer"] = "association_layer"

b_b_inh = deepcopy(inh_exc)
b_b_inh["from layer"] = "layer_b_inh"
b_b_inh["to layer"] = "layer_b"

a_self = deepcopy(exc_exc_self)
a_self["from layer"] = "layer_a"
a_self["to layer"] = "layer_a"

ass_self = deepcopy(exc_exc_self)
ass_self["from layer"] = "association_layer"
ass_self["to layer"] = "association_layer"

b_self = deepcopy(exc_exc_self)
b_self["from layer"] = "layer_b"
b_self["to layer"] = "layer_b"

a_self_inh = deepcopy(inh_inh_self)
a_self_inh["from layer"] = "layer_a_inh"
a_self_inh["to layer"] = "layer_a_inh"

ass_self_inh = deepcopy(inh_inh_self)
ass_self_inh["from layer"] = "association_layer_inh"
ass_self_inh["to layer"] = "association_layer_inh"

b_self_inh = deepcopy(inh_inh_self)
b_self_inh["from layer"] = "layer_b_inh"
b_self_inh["to layer"] = "layer_b_inh"


input_conn = {
    "type" : "one to one",
    "opcode" : "add",
    "plastic" : False,
    "max" : "1.0",
    "weight config" : {
        "type" : "flat",
        "weight" : "1.0",
        "fraction" : "1.0"
    },
    "myelinated" : True
}
gate_conn = {
    "type" : "divergent",
    "opcode" : "mult",
    "plastic" : False,
    "max" : "1.0",
    "weight config" : {
        "type" : "flat",
        "weight" : "1.0",
        "fraction" : "1.0"
    },
    "arborized config" : {
        "field size" : dim/input_grid,
        "stride" : dim/input_grid,
        "wrap" : False,
        "offset" : "0"
    },
    "myelinated" : True
}


in_a = deepcopy(input_conn)
in_a["from layer"] = "input_a"
in_a["to layer"] = "layer_a"
in_a["short term plasticity"] = False
g_a = deepcopy(gate_conn)
g_a["from layer"] = "gate_a"
g_a["to layer"] = "input_a"
g_a["short term plasticity"] = False

in_b = deepcopy(input_conn)
in_b["from layer"] = "input_b"
in_b["to layer"] = "layer_b"
in_b["short term plasticity"] = False
g_b = deepcopy(gate_conn)
g_b["from layer"] = "gate_b"
g_b["to layer"] = "input_b"
g_b["short term plasticity"] = False



# Create connections
connections = [
    in_a, g_a,
    in_b, g_b,
    a_ass, ass_a,
    b_ass, ass_b,
    a_self, ass_self, b_self,
    a_a, ass_ass, b_b,
    a_a_inh, ass_ass_inh, b_b_inh,
    #a_self_inh, ass_self_inh, b_self_inh,
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
        "window" : 1000000, # Long term
        "linear" : True,
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
        "window" : change_rate, # Short term
        "linear" : False,
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
        "random" : False,
        "value" : "1.0",
        "rate" : change_rate,
        "layers" : [
            { "structure" : "snn", "layer" : "gate_a" },
        ]
    },
    {
        "type" : "one_hot_random_input",
        "random" : False,
        "value" : "1.0",
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

print(network.run(env, {"multithreaded" : True,
                        "worker threads" : 1,
                        "devices" : get_gpus(),
                        "verbose" : True}))

# Delete the objects
del network
del env
