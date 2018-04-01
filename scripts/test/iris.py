from syngen import Network, Environment, create_io_callback, FloatArray
from ctypes import cast, POINTER, c_float

data_path = "./resources/iris/"

# Create test callback
def callback(ID, size, ptr):
    arr = FloatArray(size, ptr)
    for x in arr: print(x)

create_io_callback("iris_cb", callback)

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
        "plastic" : True,
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
        "filename" : data_path + "/iris_input.csv",
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
#                "output" : True,
#                "function" : "iris_cb",
#                "id" : 0
#            }
#        ]
#    }
#    {
#        "type" : "csv_evaluator",
#        "filename" : data_path + "/iris_output.csv",
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
    print(network.run(train_env, {"multithreaded" : False,
                                  "verbose" : True}))

    # Save the state and load it back up
    network.save_state("states/iris.bin")
else:
    network.load_state("states/iris.bin")

# Retrieve main weight matrix
matrix = network.get_weight_matrix("main matrix")
for x in matrix:
    print(x)

for i in range(som_dim * som_dim):
    print(matrix[(i*4):(i*4+4)])

# Delete the objects
del network
del train_env
