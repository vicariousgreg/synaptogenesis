from syngen import Network, Environment, create_io_callback, FloatArray

# Create main structure (feedforward engine)
structure = {"name" : "test", "type" : "parallel"}

exc = {
    "name" : "exc",
    "neural model" : "izhikevich",
    "rows" : 200,
    "columns" : 200,
    "params" : "random positive",
    "neuron spacing" : "0.1",
    "init config" : {
        "type" : "poisson",
        "rate" : 10
    },
}
inh = {
    "name" : "inh",
    "neural model" : "izhikevich",
    "rows" : 100,
    "columns" : 100,
    "params" : "random negative",
    "neuron spacing" : "0.2"
}
structure["layers"] = [exc, inh]

conn1 = {
    "from layer" : "exc",
    "to layer" : "inh",
    "type" : "convergent",
    "arborized config" : {
        "field size" : 15,
        "stride" : "2",
        "wrap" : False,
    },
    "opcode" : "add",
    "max" : "1.0",
    "plastic" : False,
    "weight config" : {
        "type" : "flat",
        "weight" : "1.0",
        "fraction" : "0.1"
    },
}

conn2 = {
    "from layer" : "inh",
    "to layer" : "exc",
    "type" : "divergent",
    "arborized config" : {
        "field size" : 15,
        "stride" : "2",
        "wrap" : False,
    },
    "opcode" : "sub",
    "plastic" : False,
    "max" : "1.0",
    "weight config" : {
        "type" : "flat",
        "weight" : "1.0",
        "fraction" : "0.1"
    },
}

conn3 = {
    "from layer" : "exc",
    "to layer" : "exc",
    "type" : "convergent",
    "arborized config" : {
        "field size" : 15,
        "wrap" : False,
    },
    "opcode" : "add",
    "max" : "1.0",
    "plastic" : False,
    "weight config" : {
        "type" : "flat",
        "weight" : "0.5",
        "fraction" : "0.1"
    },
}

# Create connections
connections = [conn1, conn2, conn3]

modules = [
    {
        "type" : "visualizer",
        "layers" : [
            { "structure" : "test", "layer" : "exc" },
            { "structure" : "test", "layer" : "inh" },
        ]
    },
#    {
#        #"type" : "one_hot_random_input",
#        "type" : "periodic_input",
#        "random" : False,
#        "value" : "1.0",
#        "rate" : "1000000",
#        "layers" : [
#            { "structure" : "test", "layer" : "exc" },
#        ]
#    },
]

env = Environment({"modules" : modules})

# Create network
network = Network(
    {"structures" : [structure],
     "connections" : connections})

print(network.run(env, {"multithreaded" : True,
                        "worker threads" : "1",
                        "refresh rate" : "100",
                        "verbose" : True}))

# Delete the objects
del network
del env
