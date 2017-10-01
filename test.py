from syngen import Network

structure = {"name" : "debug", "type" : "parallel"}

rows = 10
cols = 20

source = {
    "name" : "source",
    "neural model" : "debug",
    "rows" : rows,
    "columns" : cols}
dest = {
    "name" : "dest",
    "neural model" : "debug",
    "rows" : rows,
    "columns" : cols}

structure["layers"] = [source, dest]
connections = [
    {
        "from layer" : "source",
        "to layer" : "dest",
        "type" : "fully connected",
        "opcode" : "add",
        "plastic" : "true",
    },
    {
        "from layer" : "source",
        "to layer" : "dest",
        "type" : "subset",
        "opcode" : "add",
        "plastic" : "true",
        "subset config" : {
            "from row start" : "2",
            "from row end" : "8",
            "from column start" : "2",
            "from column end" : "18",
            "to row start" : "2",
            "to row end" : "8",
            "to column start" : "2",
            "to column end" : "18"}
    },
    {
        "from layer" : "source",
        "to layer" : "dest",
        "type" : "one to one",
        "opcode" : "add",
        "plastic" : "true",
    }
]

network = Network(
    {"structures" : [structure],
     "connections" : connections})

network.save("test.json")
network.run(1, True)

del network
