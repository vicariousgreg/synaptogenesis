from syngen import Network, Environment, create_io_callback, FloatArray, IntArray
from syngen import create_indices_weight_callback, get_cpu

# Prints weight indices
def print_indices_callback(ID, size, weights, from_indices, to_indices, used):
    w_arr = FloatArray(size, weights)
    fi_arr = IntArray(size, from_indices)
    ti_arr = IntArray(size, to_indices)
    used_arr = IntArray(size, used)

    for i in range(size):
        print(w_arr.data[i], fi_arr.data[i], ti_arr.data[i], used_arr.data[i])

create_indices_weight_callback("print", print_indices_callback)

structure = {"name" : "test", "type" : "parallel"}

rows = 100
columns = 100

a = {
    "name" : "a",
    "neural model" : "relay",
    "ramp" : True,
    "rows" : rows,
    "columns" : columns,
}
b = {
    "name" : "b",
    "neural model" : "relay",
    "ramp" : True,
    "rows" : rows,
    "columns" : columns,
}
c = {
    "name" : "c",
    "neural model" : "relay",
    "ramp" : True,
    "rows" : rows,
    "columns" : columns,
}
d = {
    "name" : "d",
    "neural model" : "relay",
    "ramp" : True,
    "rows" : rows,
    "columns" : columns,
}

structure["layers"] = [a, b, c, d]

oto_conn = {
    "from layer" : "a",
    "to layer" : "b",
    "type" : "one to one",
    "opcode" : "add",
    "max" : "1.0",
    "plastic" : False,
    "randomized projection" : True,
    "weight config" : {
        "type" : "flat",
        "weight" : "1.0",
        "fraction" : "1.0",
        "indices callback" : "print",
    }
}

conv_conn = {
    "from layer" : "a",
    "to layer" : "c",
    "type" : "convergent",
    "opcode" : "add",
    "plastic" : False,
    "randomized projection" : True,
    "arborized config" : {
        "column field size" : 3,
        "row field size" : 1,
        "stride" : 1,
        "offset" : 0,
        "wrap" : True,
    },
    "weight config" : {
        "type" : "flat",
        "weight" : 1.0 / 9,
        "indices callback" : "print",
    }
}

div_conn = {
    "from layer" : "a",
    "to layer" : "d",
    "type" : "divergent",
    "opcode" : "add",
    "plastic" : False,
    "randomized projection" : True,
    "arborized config" : {
        "column field size" : 3,
        "row field size" : 1,
        "stride" : 1,
        "offset" : 0,
        "wrap" : True,
    },
    "weight config" : {
        "type" : "flat",
        "weight" : 1.0,
        "indices callback" : "print",
    }
}

# Create connections
connections = [oto_conn, conv_conn, div_conn]
#connections = [div_conn]
#connections = [conv_conn]
#connections = [oto_conn, conv_conn]

env = Environment(
{
    "modules" : [
    {
        "type" : "visualizer",
        "layers" : [
            { "structure" : "test", "layer" : "a" },
            { "structure" : "test", "layer" : "b" },
            { "structure" : "test", "layer" : "c" },
            { "structure" : "test", "layer" : "d" },
        ]
    },
    {
        "type" : "one_hot_random_input",
        "rate" : 100,
        "layers" : [
            { "structure" : "test", "layer" : "a" },
        ]
    },
#    {
#        "type" : "gaussian_random_input",
#        "rate" : "100",
#        "border" : 0,
#        "std dev" : 3,
#        "value" : 1.0,
#        "normalize" : True,
#        "peaks" : 1,
#        "random" : False,
#        "layers" : [
#            { "structure" : "test", "layer" : "a" },
#        ]
#    }
    ]
})

# Create network
network = Network(
    {"structures" : [structure],
     "connections" : connections})

print(network.run(env, {"multithreaded" : True,
                        "iterations" : "1000000",
                        "worker threads" : "0",
                        "refresh rate" : "100",
                        "devices" : get_cpu(),
                        "verbose" : True}))

# Delete the objects
del network
del env
