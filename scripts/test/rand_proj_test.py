from syngen import Network, Environment, create_io_callback, FloatArray, IntArray
from syngen import create_indices_weight_callback, get_cpu

# Prints weight indices
def print_indices_callback(ID, size, weights, from_indices, to_indices, used):
    w_arr = FloatArray(size, weights)
    fi_arr = IntArray(size, from_indices)
    ti_arr = IntArray(size, to_indices)
    used_arr = IntArray(size, used)

    for i in xrange(size):
        print(w_arr.data[i], fi_arr.data[i], ti_arr.data[i], used_arr.data[i])

create_indices_weight_callback("print", print_indices_callback)

structure = {"name" : "test", "type" : "parallel"}

a = {
    "name" : "a",
    "neural model" : "debug",
    "ramp" : True,
    "rows" : 1,
    "columns" : 10,
}
b = {
    "name" : "b",
    "neural model" : "debug",
    "ramp" : True,
    "rows" : 1,
    "columns" : 10,
}
c = {
    "name" : "c",
    "neural model" : "debug",
    "ramp" : True,
    "rows" : 1,
    "columns" : 10,
}

structure["layers"] = [a, b, c]

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
    "from layer" : "b",
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
        "weight" : 1.0,
        "indices callback" : "print",
    }
}

div_conn = {
    "from layer" : "c",
    "to layer" : "b",
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

env = Environment({})

# Create network
network = Network(
    {"structures" : [structure],
     "connections" : connections})

print(network.run(env, {"multithreaded" : True,
                        "iterations" : "1",
                        "worker threads" : "0",
                        "refresh rate" : "100",
                        "devices" : get_cpu(),
                        "verbose" : True}))

# Delete the objects
del network
del env
