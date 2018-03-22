from syngen import Network, Environment, create_io_callback, FloatArray

# Create main structure (feedforward engine)
structure = {"name" : "test", "type" : "parallel"}

a = {
    "name" : "a",
    "neural model" : "relay",
    "rows" : 1,
    "columns" : 1,
    "init config" : {
        "type" : "normal",
        "mean" : 1.0,
        "std dev" : 0.0
    },
}
b = {
    "name" : "b",
    "neural model" : "relay",
    "rows" : 64,
    "columns" : 64
}
c = {
    "name" : "c",
    "neural model" : "relay",
    "rows" : 2,
    "columns" : 2
}

structure["layers"] = [a, b, c]

conn1 = {
    "from layer" : "a",
    "to layer" : "b",
    "type" : "divergent",
    "arborized config" : {
        "field size" : 8,
        "spacing" : 8,
        "stride" : 1,
        "wrap" : "false",
    },
    "opcode" : "add",
    "max" : "1.0",
    "plastic" : "false",
    "weight config" : {
        "type" : "flat",
        "weight" : "1.0",
        "fraction" : "1.0"
    }
}

conn2 = {
    "from layer" : "b",
    "to layer" : "c",
    "type" : "convergent",
    "arborized config" : {
        "field size" : 8,
        "spacing" : 8,
        "stride" : 1,
        "wrap" : "false",
    },
    "opcode" : "add",
    "max" : "1.0",
    "plastic" : "false",
    "weight config" : {
        "type" : "flat",
        "weight" : "1.0",
        "fraction" : "1.0"
    }
}

# Create connections
connections = [conn1, conn2]

modules = [
    {
        "type" : "visualizer",
        "layers" : [
            { "structure" : "test", "layer" : "a" },
            { "structure" : "test", "layer" : "b" },
            { "structure" : "test", "layer" : "c" },
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
                        "refresh rate" : "100",
                        "verbose" : "true"}))

# Delete the objects
del network
del env
