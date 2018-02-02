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
            "second order" : "true"
        },
        {
            "name" : "right",
            "second order" : "true"
        },
    ]}
in1 = {
    "name" : "in1",
    "neural model" : "relay",
    "rows" : dim,
    "columns" : dim}
in2 = {
    "name" : "in2",
    "neural model" : "relay",
    "rows" : dim,
    "columns" : dim}
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
        "plastic" : "false",
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
        "plastic" : "false",
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
        "plastic" : "false",
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
        "plastic" : "false",
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
        "type" : "periodic_input",
        "rate" : "1",
        "random" : "true",
        "layers" : [
            {
                "structure" : "dendrite_test",
                "layer" : "in1"
            }
        ]
    },
    {
        "type" : "periodic_input",
        "rate" : "1",
        "random" : "true",
        "layers" : [
            {
                "structure" : "dendrite_test",
                "layer" : "in2"
            }
        ]
    },
    {
        "type" : "gaussian_random_input",
        "rate" : "100",
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
print(network.run(env, {"multithreaded" : "true",
                        "worker threads" : 1,
                        "devices" : device,
                        "learning flag" : "false"}))

# Delete the objects
del network
del env
