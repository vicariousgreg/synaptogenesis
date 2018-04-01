from syngen import Network, Environment
from syngen import get_gpus, get_cpu

# Create main structure (parallel engine)
structure = {"name" : "dendrite_test", "type" : "parallel"}

# Add layers (represent bias as a layer)
dim = 100
receiver = {
    "name" : "receiver",
    "neural model" : "relay",
    "rows" : dim,
    "columns" : dim,
    "dendrites" : [
        {
            "name" : "left",
            "second order" : True
        },
        {
            "name" : "right",
            "second order" : True
        },
    ]}
in1 = {
    "name" : "in1",
    "neural model" : "relay",
    "rows" : dim,
    "columns" : dim,
    "init config" : {
        "type" : "uniform",
        "min" : 0.0,
        "max" : 1.0,
    }}
in2 = {
    "name" : "in2",
    "neural model" : "relay",
    "rows" : dim,
    "columns" : dim,
    "init config" : {
        "type" : "uniform",
        "min" : 0.0,
        "max" : 0.5,
    }}
gate1 = {
    "name" : "gate1",
    "neural model" : "relay",
    "rows" : dim,
    "columns" : dim}
gate2 = {
    "name" : "gate2",
    "neural model" : "relay",
    "rows" : dim,
    "columns" : dim}

# Add layers to structure
structure["layers"] = [receiver, in1, in2, gate1, gate2]

# Create connections
connections = [
    {
        "from layer" : "in1",
        "to layer" : "receiver",
        "dendrite" : "left",
        "type" : "one to one",
        "opcode" : "add",
        "plastic" : False,
        "weight config" : {
            "type" : "flat",
            "weight" : "1"
        }
    },
    {
        "from layer" : "gate1",
        "to layer" : "receiver",
        "dendrite" : "left",
        "type" : "one to one",
        "opcode" : "mult",
        "plastic" : False,
        "weight config" : {
            "type" : "flat",
            "weight" : "1"
        }
    },
    {
        "from layer" : "in2",
        "to layer" : "receiver",
        "dendrite" : "right",
        "type" : "one to one",
        "opcode" : "add",
        "plastic" : False,
        "weight config" : {
            "type" : "flat",
            "weight" : "1"
        }
    },
    {
        "from layer" : "gate2",
        "to layer" : "receiver",
        "dendrite" : "right",
        "type" : "one to one",
        "opcode" : "mult",
        "plastic" : False,
        "weight config" : {
            "type" : "flat",
            "weight" : "1"
        }
    },
]

# Create environment modules
modules = [
    {
        "type" : "visualizer",
        "layers" : [
            { "structure" : "dendrite_test", "layer" : "receiver" },
            { "structure" : "dendrite_test", "layer" : "in1" },
            { "structure" : "dendrite_test", "layer" : "gate1" },
            { "structure" : "dendrite_test", "layer" : "in2" },
            { "structure" : "dendrite_test", "layer" : "gate2" },
        ]
    },
    {
        "type" : "gaussian_random_input",
        "rate" : "100",
        "std dev" : "10",
        "value" : "1.0",
        "normalize" : True,
        "peaks" : "1",
        "random" : False,
        "layers" : [
            {
                "structure" : "dendrite_test",
                "layer" : "gate1"
            }
        ]
    },
    {
        "type" : "gaussian_random_input",
        "rate" : "100",
        "std dev" : "10",
        "value" : "1.0",
        "normalize" : True,
        "peaks" : "1",
        "random" : False,
        "layers" : [
            {
                "structure" : "dendrite_test",
                "layer" : "gate2"
            }
        ]
    },
]

env = Environment({"modules" : modules})

# Create network
network = Network(
    {"structures" : [structure],
     "connections" : connections})

# Run test
gpus = get_gpus()
device = gpus[len(gpus)-1] if len(gpus) > 0 else get_cpu()
print(network.run(env, {"multithreaded" : True,
                        "worker threads" : 1,
                        "devices" : device,
                        "learning flag" : False}))

# Delete the objects
del network
del env
