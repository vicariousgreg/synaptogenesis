from syngen import Network, Environment
from syngen import get_gpus, get_cpu
from math import exp

def gaussian_weight_string(rows, cols, peak, sig, norm=False):
    inv_sqrt_2pi = 0.3989422804014327

    row_center = rows / 2
    col_center = cols / 2
    peak_coeff = peak * (1.0 if norm else inv_sqrt_2pi / sig)
    gauss = []

    for row in xrange(rows):
        for col in xrange(cols):
            dist = ((row - row_center) ** 2) + ((col - col_center) ** 2) ** 0.5
            a = dist / sig
            gauss.append("%.6f" % (peak_coeff * exp(-0.5 * a * a)))
    return " ".join(gauss)

# Create main structure (parallel engine)
structure = {"name" : "oculomotor", "type" : "parallel"}

# Add layers (represent bias as a layer)
vision_layer = {
    "name" : "vision",
    "neural model" : "relay",
    "rows" : 100,
    "columns" : 100}
sc_layer = {
    "name" : "sc",
    "neural model" : "relay",
    "rows" : 100,
    "columns" : 100}

# Add layers to structure
structure["layers"] = [vision_layer, sc_layer]

# Create connections
receptive_field = 15
connections = [
    {
        "from layer" : "vision",
        "to layer" : "sc",
        "type" : "convergent",
        "convolutional" : "true",
        "opcode" : "add",
        "plastic" : "false",
        "weight config" : {
            "type" : "specified",
            "weight string" : gaussian_weight_string(
                receptive_field, receptive_field, 3, 3, False)
        },
        "arborized config" : {
            "field size" : receptive_field,
            "stride" : 1,
        }
    },
]

# Create environment modules
modules = [
    {
        "type" : "visualizer",
        "layers" : [
            { "structure" : "oculomotor", "layer" : "vision" },
            { "structure" : "oculomotor", "layer" : "sc" }
        ]
    },
    {
        "type" : "gaussian_random_input",
        "rate" : "100",
        "std dev" : "10",
        "value" : "1",
        "normalize" : "false",
        "peaks" : "3",
        "layers" : [
            {
                "structure" : "oculomotor",
                "layer" : "vision"
            }
        ]
    }
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
                        "refresh rate" : 100,
                        "learning flag" : "false"}))

# Delete the objects
del network
del env
