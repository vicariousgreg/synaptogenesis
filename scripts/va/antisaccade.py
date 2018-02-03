from syngen import Network, Environment
from syngen import get_gpus, get_cpu

# Create main structure (parallel engine)
structure = {"name" : "saccade", "type" : "parallel"}

# Add layers (represent bias as a layer)
vision_layer = {
    "name" : "vision",
    "neural model" : "relay",
    "rows" : 340,
    "columns" : 480}

# Add layers to structure
structure["layers"] = [vision_layer]

# Create environment modules
modules = [
    {
        "type" : "saccade",
        "layers" : [
            {
                "structure" : "saccade",
                "layer" : "vision",
                "params" : "input",
            }
        ]
    },
    {
        "type" : "visualizer",
        "layers" : [
            { "structure" : "saccade", "layer" : "vision" }
        ]
    },
]

env = Environment({"modules" : modules})

# Create network
network = Network(
    {"structures" : [structure],
     "connections" : []})

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
