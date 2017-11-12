from syngen import Network, Environment, get_cpu, get_gpus

data_path = "/HDD/datasets/mnist/processed/"
# data_path = "./resources"

# Create main structure (feedforward engine)
structure = {"name" : "mnist", "type" : "feedforward"}

# Add layers (represent bias as a layer)
input_layer = {
    "name" : "input_layer",
    "neural model" : "relay",
    "rows" : 28,
    "columns" : 28}
output_layer = {
    "name" : "output_layer",
    "neural model" : "perceptron",
    "rows" : 1,
    "columns" : 10}
bias_layer = {
    "name" : "bias_layer",
    "neural model" : "relay",
    "rows" : 1,
    "columns" : 1}

# Add layers to structure
structure["layers"] = [input_layer, output_layer, bias_layer]

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
            "type" : "flat",
            "weight" : "0"
        }
    },
    {
        "name" : "bias matrix",
        "from layer" : "bias_layer",
        "to layer" : "output_layer",
        "type" : "fully connected",
        "opcode" : "add",
        "plastic" : "true",
        "weight config" : {
            "type" : "flat",
            "weight" : "0"
        }
    }
]

# Create environment modules
# Loads training MNIST CSV files
# One for input, one for expected output
modules = [
    {
        "type" : "csv_input",
        "filename" : data_path + "/mnist_train_input.csv",
        "offset" : 0,
        "exposure" : 1,
        "normalization" : "255",
        "layers" : [
            {
                "structure" : "mnist",
                "layer" : "input_layer"
            }
        ]
    },
    {
        "type" : "csv_evaluator",
        "filename" : data_path + "/mnist_train_output.csv",
        "offset" : 0,
        "exposure" : 1,
        "normalization" : "1",
        "layers" : [
            {
                "structure" : "mnist",
                "layer" : "output_layer"
            }
        ]
    },
    {
        "type" : "periodic_input",
        "val" : "1",
        "layers" : [
            {
                "structure" : "mnist",
                "layer" : "bias_layer"
            }
        ]
    }
]

# Create training environment
train_env = Environment({"modules" : modules})

# Replace files with test set and create new test environment
modules[0]["filename"] = data_path + "/mnist_test_input.csv";
modules[1]["filename"] = data_path + "/mnist_test_output.csv";
test_env = Environment({"modules" : modules})

# Create network
network = Network(
    {"structures" : [structure],
     "connections" : connections})

train = True
if (train):
    # Run training
    print(network.run(train_env, {"multithreaded" : "false",
                                  "devices" : get_cpu(),
                                  "worker threads" : 0}))

    # Save the state and load it back up
    network.save_state("mnist.bin")
else:
    network.load_state("mnist.bin")

# Retrieve main weight matrix
matrix = network.get_weight_matrix("main matrix")

# Run test
print(network.run(test_env, {"multithreaded" : "false",
                             "devices" : get_gpus()[0],
                             "worker threads" : 2,
                             "learning flag" : "false"}))

# Delete the objects
del network
del train_env
del test_env
