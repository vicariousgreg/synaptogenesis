from syngen import Network, Environment, create_callback, FloatArray

# Create main structure (feedforward engine)
structure = {"name" : "snn", "type" : "parallel"}

# Add layers (represent bias as a layer)
dim = 64

layer = {
    "neural model" : "izhikevich",
    "rows" : dim,
    "columns" : dim,
    "noise config" : {
        "type" : "poisson",
        "rate" : 0.01
    },
    "neuron spacing" : "0.1",
    "init" : "random positive"
}

input_layer = dict(layer)
input_layer["name"] = "input_layer"

association_layer = dict(layer)
association_layer["name"] = "association_layer"
association_layer["rows"] = dim*2
association_layer["columns"] = dim*2
association_layer["neuron spacing"] = "0.05"
del association_layer["noise config"]

output_layer = dict(layer)
output_layer["name"] = "output_layer"

layer_inh = {
    "neural model" : "izhikevich",
    "rows" : dim/4,
    "columns" : dim/4,
    "neuron spacing" : "0.2",
    "init" : "random negative"
}

input_layer_inh = dict(layer_inh)
input_layer_inh["name"] = "input_layer_inh"

association_layer_inh = dict(layer_inh)
association_layer_inh["name"] = "association_layer_inh"
association_layer_inh["rows"] = dim/2
association_layer_inh["columns"] = dim/2
association_layer_inh["neuron spacing"] = "0.1"

output_layer_inh = dict(layer_inh)
output_layer_inh["name"] = "output_layer_inh"

# Add layers to structure
structure["layers"] = [
    input_layer, association_layer, output_layer,
    input_layer_inh, association_layer_inh, output_layer_inh]

exc_exc = {
    "type" : "convergent",
    "arborized config" : {
        "field size" : 31,
        "wrap" : "true",
    },
    "opcode" : "add",
    "plastic" : "true",
    "max" : "0.5",
    "weight config" : {
        "type" : "power law",
        "exponent" : "2.5",
        "fraction" : "0.1"
    },
}
exc_exc_self = {
    "type" : "convergent",
    "arborized config" : {
        "field size" : 31,
        "wrap" : "true",
    },
    "opcode" : "add",
    "plastic" : "true",
    "max" : "0.5",
    "weight config" : {
        "type" : "power law",
        "exponent" : "2.5",
        "fraction" : "0.1"
    },
}
exc_inh = {
    "type" : "convergent",
    "arborized config" : {
        "field size" : 15,
        "stride" : 2,
        "wrap" : "true",
    },
    "opcode" : "add",
    "plastic" : "false",
    "max" : "0.5",
    "weight config" : {
        "type" : "power law",
        "exponent" : "1.5",
        "fraction" : "0.1"
    },
}
inh_exc = {
    "type" : "divergent",
    "arborized config" : {
        "field size" : 15,
        "stride" : 2,
        "wrap" : "true",
    },
    "opcode" : "add",
    "plastic" : "true",
    "weight config" : {
        "type" : "flat",
        "weight" : "0.1",
        "fraction" : "0.1"
    },
}

a = dict(exc_exc)
a["from layer"] = "input_layer"
a["to layer"] = "association_layer"
a["type"] = "divergent"
a["arborized config"]["stride"] = "2"
b = dict(exc_exc)
b["from layer"] = "association_layer"
b["to layer"] = "output_layer"
b["arborized config"]["stride"] = "2"
c = dict(exc_exc)
c["from layer"] = "output_layer"
c["to layer"] = "association_layer"
c["type"] = "divergent"
c["arborized config"]["stride"] = "2"
d = dict(exc_exc)
d["from layer"] = "association_layer"
d["to layer"] = "input_layer"
d["arborized config"]["stride"] = "2"

in_in = dict(exc_inh)
in_in["from layer"] = "input_layer"
in_in["to layer"] = "input_layer_inh"

ass_ass = dict(exc_inh)
ass_ass["from layer"] = "association_layer"
ass_ass["to layer"] = "association_layer_inh"

out_out = dict(exc_inh)
out_out["from layer"] = "output_layer"
out_out["to layer"] = "output_layer_inh"

in_in_inh = dict(inh_exc)
in_in_inh["from layer"] = "input_layer_inh"
in_in_inh["to layer"] = "input_layer"

ass_ass_inh = dict(inh_exc)
ass_ass_inh["from layer"] = "association_layer_inh"
ass_ass_inh["to layer"] = "association_layer"

out_out_inh = dict(inh_exc)
out_out_inh["from layer"] = "output_layer_inh"
out_out_inh["to layer"] = "output_layer"

in_self = dict(exc_exc_self)
in_self["from layer"] = "input_layer"
in_self["to layer"] = "input_layer"

ass_self = dict(exc_exc_self)
ass_self["from layer"] = "association_layer"
ass_self["to layer"] = "association_layer"

out_self = dict(exc_exc_self)
out_self["from layer"] = "output_layer"
out_self["to layer"] = "output_layer"

# Create connections
connections = [
    a, b, c, d,
    in_self, ass_self, out_self,
    in_in, ass_ass, out_out,
    in_in_inh, ass_ass_inh, out_out_inh
]

modules = [
    {
        "type" : "visualizer",
        "layers" : [
            { "structure" : "snn", "layer" : "input_layer" },
            { "structure" : "snn", "layer" : "association_layer" },
            { "structure" : "snn", "layer" : "output_layer" },
            { "structure" : "snn", "layer" : "input_layer_inh" },
            { "structure" : "snn", "layer" : "association_layer_inh" },
            { "structure" : "snn", "layer" : "output_layer_inh" },
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
