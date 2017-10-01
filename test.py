from syngen import Network, Environment

structure = {"name" : "mnist", "type" : "feedforward"}

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

structure["layers"] = [input_layer, output_layer, bias_layer]
connections = [
    {
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

modules = [
    {
        "type" : "csv_input",
        "filename" : "/HDD/datasets/mnist/processed/mnist_train_input.csv",
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
        "filename" : "/HDD/datasets/mnist/processed/mnist_train_output.csv",
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

train_env = Environment({"modules" : modules})

modules[0]["filename"] = "/HDD/datasets/mnist/processed/mnist_test_input.csv";
modules[1]["filename"] = "/HDD/datasets/mnist/processed/mnist_test_output.csv";
test_env = Environment({"modules" : modules})

network = Network(
    {"structures" : [structure],
     "connections" : connections})

network.run(train_env,
    {"iterations" : "60000",
     "verbose" : "true"})
network.run(test_env,
    {"iterations" : "10000",
     "verbose" : "true",
     "learning flag" : "false"})

del network
