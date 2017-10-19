from syngen import Network, Environment

# Create main structure (parallel engine)
structure = {"name" : "dsst", "type" : "parallel"}

# DSST parameters
def get_dsst_params(num_rows=8, cell_res=8):
    num_cols = 18
    cell_rows = 2*cell_res + 1
    spacing = cell_res / 4

    input_rows = (num_rows + 2) * (cell_rows + spacing) - spacing
    input_cols = num_cols * (cell_res + spacing) - spacing

    focus_rows = input_rows - cell_rows
    focus_cols = input_cols - cell_res

    return {
        "columns" : num_cols,
        "rows" : num_rows,
        "cell columns" : cell_res,
        "cell rows" : cell_rows,
        "spacing" : spacing,
        "input rows" : input_rows,
        "input columns" : input_cols,
        "focus rows" : focus_rows,
        "focus columns" : focus_cols,
    }

dsst_params = get_dsst_params(num_rows=8, cell_res=8)


# Add layers (represent bias as a layer)
vision_layer = {
    "name" : "vision",
    "neural model" : "relay",
    "rows" : dsst_params["input rows"],
    "columns" : dsst_params["input columns"]}
what_layer = {
    "name" : "what",
    "neural model" : "relay",
    "rows" : dsst_params["cell rows"],
    "columns" : dsst_params["cell columns"],
    "dendrites" : [
        {
            "name" : "fixation",
            "second order" : "true"
        }
    ]}
focus_layer = {
    "name" : "focus",
    "neural model" : "relay",
    "rows" : dsst_params["focus rows"],
    "columns" : dsst_params["focus columns"]}

# Add layers to structure
structure["layers"] = [vision_layer, what_layer, focus_layer]

# Create connections
connections = [
    {
        "from layer" : "vision",
        "to layer" : "what",
        "dendrite" : "fixation",
        "type" : "convergent",
        "convolutional" : "true",
        "opcode" : "add",
        "plastic" : "false",
        "weight config" : {
            "type" : "flat",
            "weight" : "1"
        },
        "arborized config" : {
            "row field size" : dsst_params["focus rows"],
            "column field size" : dsst_params["focus columns"],
            "offset" : "0"
        }
    },
    {
        "from layer" : "focus",
        "to layer" : "what",
        "dendrite" : "fixation",
        "type" : "convergent",
        "convolutional" : "true",
        "opcode" : "mult",
        "plastic" : "false",
        "weight config" : {
            "type" : "flat",
            "weight" : "1"
        },
        "arborized config" : {
            "row field size" : dsst_params["focus rows"],
            "column field size" : dsst_params["focus columns"],
            "stride" : "0",
            "offset" : "0"
        }
    }
]

# Create environment modules
modules = [
    {
        "type" : "dsst",
        "rows" : dsst_params["rows"],
        "columns" : dsst_params["columns"],
        "cell size" : dsst_params["cell columns"],
        "layers" : [
            {
                "structure" : "dsst",
                "layer" : "vision",
                "params" : "input",
            }
        ]
    },
    {
        "type" : "visualizer",
        "layers" : [
            { "structure" : "dsst", "layer" : "vision" },
            { "structure" : "dsst", "layer" : "what" },
            { "structure" : "dsst", "layer" : "focus" }
        ]
    },
    {
        "type" : "one_hot_random_input",
        "rate" : "100",
        "layers" : [
            {
                "structure" : "dsst",
                "layer" : "focus"
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
print(network.run(env, {"multithreaded" : "true",
                        "worker threads" : 1,
                        "learning flag" : "false"}))

# Delete the objects
del network
del env
