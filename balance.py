from syngen import Network, Environment, create_callback, FloatArray
from syngen import get_gpus, get_cpu
from syngen import set_suppress_output, set_warnings, set_debug
from os import path

leaky = False
visualizer = True

set_suppress_output(False)
set_warnings(False)
set_debug(False)

# Create main structure (feedforward engine)
structure = {"name" : "snn", "type" : "parallel"}

dim = 128
exc_exc_spread = 21
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
    #"init" : "random positive"
}

# Inhibitory layer
inhibitory = {
    "name" : "inh",
    "neural model" : ("leaky_izhikevich" if leaky else "izhikevich"),
    "rows" : dim/2,
    "columns" : dim/2,
    "noise config" : {
        "type" : "poisson",
        "value" : 10.0,
        "rate" : 100
    },
    "neuron spacing" : "0.2",
    "init" : "fast"
    #"init" : "random negative"
}

# Add layers to structure
structure["layers"] = [excitatory, inhibitory]

exc_exc = {
    "from layer" : "exc",
    "to layer" : "exc",
    "name" : "exc matrix",
    "type" : "convergent",
    "arborized config" : {
        "field size" : exc_exc_spread,
        "wrap" : "true",
    },
    "opcode" : "add",
    "plastic" : "true",
#    "learning rate" : "0.1",
    "max weight" : "0.5",
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
    "plastic" : "true",
    "max weight" : "1.0",
    "weight config" : {
#        "type" : "power law",
#        "exponent" : "1.5",
        "type" : "flat",
        "weight" : "0.2",
        "fraction" : "0.1"
    },
}
inh_exc = {
    "from layer" : "inh",
    "to layer" : "exc",
    "name" : "inh matrix",
    "type" : "divergent",
    "arborized config" : {
        "field size" : inh_exc_spread,
        "stride" : 2,
        "wrap" : "true",
    },
    "opcode" : "sub",
    "plastic" : "true",
    "max weight" : "5.0",
    "weight config" : {
#        "type" : "power law",
#        "exponent" : "1.5",
        "type" : "flat",
        "weight" : "0.3",
        "fraction" : "0.3"
    },
    "myelinated" : "false"
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
    "plastic" : "true",
    "max weight" : "0.5",
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

pre_exc_matrix = network.get_weight_matrix("exc matrix").to_list()
pre_inh_matrix = network.get_weight_matrix("inh matrix").to_list()

if not path.exists("./states/" + state_path):
    gpus = get_gpus()
    if len(gpus) > 1:
        device = gpus[0 if leaky else 1]
    elif len(gpus) == 1:
        device = gpus[0]
    else:
        device = get_cpu()
    print(network.run(env, {"multithreaded" : "true",
                            "worker threads" : "1",
                            "devices" : device,
                            "iterations" : 1000000,
                            "verbose" : "true"}))
    network.save_state(state_path)
else:
    network.load_state(state_path)

post_exc_matrix = network.get_weight_matrix("exc matrix").to_list()
post_inh_matrix = network.get_weight_matrix("inh matrix").to_list()

print("Excitatory Matrix:")
print("Pre:",
    "%9d" % sum(1 for x in pre_exc_matrix if x > 0.0),
    "%9.7f" % (sum(pre_exc_matrix) / sum(1 for x in pre_exc_matrix if x > 0.0)),
    "%9.7f" % sum(pre_exc_matrix),
    "%9.7f" % min(x for x in pre_exc_matrix if x > 0.0),
    "%9.7f" % max(pre_exc_matrix))
print("Post:",
    "%9d" % sum(1 for x in post_exc_matrix if x > 0.0),
    "%9.7f" % (sum(post_exc_matrix) / sum(1 for x in post_exc_matrix if x > 0.0)),
    "%9.7f" % sum(post_exc_matrix),
    "%9.7f" % min(x for x in post_exc_matrix if x > 0.0),
    "%9.7f" % max(post_exc_matrix))
diff = sum(post_exc_matrix) - sum(pre_exc_matrix)

count = sum(1 for x in pre_exc_matrix if x > 0.0)
print("Diff:", diff, count, diff / count)

print("\nInhibitory Matrix:")
print("Pre:",
    "%9d" % sum(1 for x in pre_inh_matrix if x > 0.0),
    "%9.7f" % (sum(pre_inh_matrix) / sum(1 for x in pre_inh_matrix if x > 0.0)),
    "%9.7f" % sum(pre_inh_matrix),
    "%9.7f" % min(x for x in pre_inh_matrix if x > 0.0),
    "%9.7f" % max(pre_inh_matrix))
print("Post:",
    "%9d" % sum(1 for x in post_inh_matrix if x > 0.0),
    "%9.7f" % (sum(post_inh_matrix) / sum(1 for x in post_inh_matrix if x > 0.0)),
    "%9.7f" % sum(post_inh_matrix),
    "%9.7f" % min(x for x in post_inh_matrix if x > 0.0),
    "%9.7f" % max(post_inh_matrix))
diff = sum(post_inh_matrix) - sum(pre_inh_matrix)

count = sum(1 for x in pre_inh_matrix if x > 0.0)
print("Diff:", diff, count, diff / count)

# Delete the objects
del network
del env
