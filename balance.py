from syngen import Network, Environment, create_callback, FloatArray
from syngen import get_num_gpus, set_cpu, set_gpu, set_multi_gpu, set_all_devices
from syngen import set_suppress_output, set_warnings, set_debug
from os import path

leaky = False
visualizer = False

if get_num_gpus() > 0:
    set_gpu(0 if leaky else 1)

set_suppress_output(False)
set_warnings(False)
set_debug(False)

# Create main structure (feedforward engine)
structure = {"name" : "snn", "type" : "parallel"}

dim = 64
exc_exc_spread = 31
inh_inh_spread = 7
exc_inh_spread = 21
inh_exc_spread = 11

# Excitatory layer
excitatory = {
    "name" : "exc",
    "neural model" : ("leaky_izhikevich" if leaky else "izhikevich"),
    "rows" : dim,
    "columns" : dim,
    "noise config" : {
        "type" : "poisson",
        "value" : 10.0,
        "rate" : 100
    },
    "neuron spacing" : "0.1",
    "init" : "regular"
}

# Inhibitory layer
inhibitory = {
    "name" : "inh",
    "neural model" : ("leaky_izhikevich" if leaky else "izhikevich"),
    "rows" : dim/2,
    "columns" : dim/2,
    "neuron spacing" : "0.2",
    "init" : "fast"
}

# Add layers to structure
structure["layers"] = [excitatory, inhibitory]

exc_exc = {
    "from layer" : "exc",
    "to layer" : "exc",
    "name" : "main matrix",
    "type" : "convergent",
    "arborized config" : {
        "field size" : exc_exc_spread,
        "wrap" : "true",
    },
    "opcode" : "add",
    "plastic" : "true",
#    "learning rate" : "0.1",
    "max" : "0.5",
    "weight config" : {
#        "type" : "power law",
#        "exponent" : "1.5",
        "type" : "flat",
        "weight" : "0.1",
        "fraction" : "0.1"
    },
}

exc_inh = {
    "from layer" : "exc",
    "to layer" : "inh",
    "type" : "convergent",
    "arborized config" : {
        "field size" : exc_inh_spread,
        "stride" : 2,
        "wrap" : "true",
    },
    "opcode" : "add",
    "plastic" : "false",
    "max" : "0.5",
    "weight config" : {
        "type" : "flat",
        "weight" : "0.2",
        "fraction" : "0.1"
    },
}
inh_exc = {
    "from layer" : "inh",
    "to layer" : "exc",
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
inh_inh = {
    "from layer" : "inh",
    "to layer" : "inh",
    "type" : "convergent",
    "arborized config" : {
        "field size" : inh_inh_spread,
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

# Create connections
connections = [
    exc_exc,
    exc_inh,
    inh_exc,
#    inh_inh
]

modules = []
if visualizer:
    modules = [
        {
            "type" : "visualizer",
            "layers" : [
                { "structure" : "snn", "layer" : "exc" },
                { "structure" : "snn", "layer" : "inh" },
            ]
        },
        {
            "type" : "heatmap",
            "window" : 1000000, # Long term
            "linear" : "true",
            "layers" : [
                { "structure" : "snn", "layer" : "exc" },
                { "structure" : "snn", "layer" : "inh" },
            ]
        },
        {
            "type" : "heatmap",
            "window" : 1000, # Short term
            "linear" : "false",
            "layers" : [
                { "structure" : "snn", "layer" : "exc" },
                { "structure" : "snn", "layer" : "inh" },
            ]
        },
    ]

env = Environment({"modules" : modules})

# Create network
network = Network(
    {"structures" : [structure],
     "connections" : connections})

state_path = ("balance_leaky.bin" if leaky else "balance.bin")

pre_matrix = network.get_weight_matrix("main matrix").to_list()
if not path.exists("./states/" + state_path):
    print(network.run(env, {"multithreaded" : "true",
                            "worker threads" : "1",
                            "iterations" : 100000,
                            "verbose" : "true"}))
    network.save_state(state_path)
else:
    network.load_state(state_path)

post_matrix = network.get_weight_matrix("main matrix").to_list()

print("Pre:", len(pre_matrix), sum(pre_matrix), min(pre_matrix), max(pre_matrix))
print("Post:", len(post_matrix), sum(post_matrix), min(post_matrix), max(post_matrix))
diff = sum(post_matrix) - sum(pre_matrix)

count = sum(1 for x in pre_matrix if x > 0.0)
print("Diff:", diff, count, diff / count)

# Delete the objects
del network
del env
