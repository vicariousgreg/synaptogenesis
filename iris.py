from syngen import Network, Environment, create_callback
from ctypes import cast, POINTER, c_float

# Create test callback
def callback(ID, size, ptr):
    vals = [float(y) for y in cast(ptr, POINTER(c_float))[:size]]
    print(" ".join(("%6.4f" % x) for x in vals))
    print(max(vals), vals.index(max(vals)))

cb,addr = create_callback(callback)

# Create main structure (feedforward engine)
structure = {"name" : "iris", "type" : "feedforward"}

# Add layers (represent bias as a layer)
som_dim = 5
input_layer = {
    "name" : "input_layer",
    "neural model" : "relay",
    "rows" : 1,
    "columns" : 4}
output_layer = {
    "name" : "output_layer",
    "neural model" : "som",
    "rows" : som_dim,
    "columns" : som_dim,
    "rbf scale" : "10"}

# Add layers to structure
structure["layers"] = [input_layer, output_layer]

# Create connections
connections = [
    {
        "name" : "main matrix",
        "from layer" : "input_layer",
        "to layer" : "output_layer",
        "type" : "fully connected",
        "opcode" : "add",
        "plastic" : "true",
        "weight config" : {
            "type" : "random",
            "weight" : "1.0"
        },
        "learning rate" : "0.01",
        "neighbor learning rate" : "0.001",
        "neighborhood size" : "2"
    }
]

# Create environment modules
# Loads training MNIST CSV files
# One for input, one for expected output
modules = [
    {
        "type" : "csv_input",
        "filename" : "/HDD/datasets/iris/iris_input.csv",
        "offset" : 0,
        "exposure" : 1,
        "normalization" : 8,
        "epochs" : 100,
        "layers" : [
            {
                "structure" : "iris",
                "layer" : "input_layer"
            }
        ]
    },
#    {
#        "type" : "callback",
#        "layers" : [
#            {
#                "structure" : "iris",
#                "layer" : "output_layer",
#                "params" : "output",
#                "function" : addr,
#                "id" : 0
#            }
#        ]
#    }
#    {
#        "type" : "csv_evaluator",
#        "filename" : "/HDD/datasets/iris/iris_output.csv",
#        "offset" : 0,
#        "exposure" : 1,
#        "epochs" : 1000,
#        "normalization" : "1",
#        "layers" : [
#            {
#                "structure" : "iris",
#                "layer" : "output_layer"
#            }
#        ]
#    }
]

# Create training environment
train_env = Environment({"modules" : modules})

# Create network
network = Network(
    {"structures" : [structure],
     "connections" : connections})

train = True
if (train):
    # Run training
    print(network.run(train_env, {"multithreaded" : "false",
                                  "verbose" : "true"}))

    # Save the state and load it back up
    network.save_state("iris.bin")
else:
    network.load_state("iris.bin")

# Retrieve main weight matrix
matrix = network.get_weight_matrix("main matrix")
for i in range(som_dim * som_dim):
    print(matrix.data[(i*4):(i*4+4)])

# Delete the objects
del network
del train_env
