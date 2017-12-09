from syngen import Network, Environment

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
print(network.run(env, {"multithreaded" : "true",
                        "worker threads" : 1,
                        "learning flag" : "false"}))

# Delete the objects
del network
del env
