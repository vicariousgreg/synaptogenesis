import subprocess
import networkx as nx
import matplotlib.pyplot as plt

command = 'find src/core'
output = subprocess.check_output(command.split(" "))
output = output.split("\n")

nodes = set()
for line in output:
    if "/" in line:
        f = line.split("/")[-1]
        if not f.startswith(".") and "." in f:
            nodes.add(f)
nodes = list(nodes)

command = 'git grep #include.*/.*.h'
output = subprocess.check_output(command.split(" "))
output = output.split("\n")

edges = set()
in_degree = dict()
out_degree = dict()

for line in output:
    if ":" not in line: continue
    from_file, to_file = line.split(":")
    from_file = from_file.split("/")[-1]
    to_file = to_file.split("/")[-1][:-1]

    if from_file in nodes and to_file in nodes:
        edges.add((from_file, to_file))

        out_degree[from_file] = out_degree.get(from_file, 0) + 1
        in_degree[to_file] = in_degree.get(to_file, 0) + 1

for node in nodes:
    if node not in in_degree:
        in_degree[node] = 0
    if node not in out_degree:
        out_degree[node] = 0

for node in reversed(sorted(list(nodes), key=lambda node: in_degree[node])):
    print(node, in_degree[node], out_degree[node])

def get_dependency_spans(reverse=False):
    spans = dict()
    in_edges = dict()
    out_edges = dict()

    for node in nodes:
        in_edges[node] = []
        out_degree[node] = 0
        spans[node] = 1

    for f,t in edges:
        if reverse:
            in_edges[f].append((t,f))
            out_degree[t] += 1
        else:
            in_edges[t].append((f,t))
            out_degree[f] += 1

    def visit(node):
        for f,t in in_edges[node]:
            spans[f] += spans[t]
            out_degree[f] -= 1

    visited = 0
    while visited < len(nodes):
        old_visited = visited
        for node in nodes + nodes:
            if out_degree[node] == 0:
                visit(node)
                visited += 1

        if visited == old_visited:
            print(visited, old_visited)
            foo

    return spans

cost = get_dependency_spans()
weight = get_dependency_spans(reverse=True)

in_edges = dict()
out_edges = dict()

for node in nodes:
    in_edges[node] = []
    out_edges[node] = []

for f,t in edges:
    in_edges[t].append((f,t))
    out_edges[f].append((f,t))

print("")
print("")
for node in reversed(sorted(list(nodes), key=lambda node: weight[node])):
    deps = reversed(sorted([(cost[t], t) for f,t in out_edges[node]],
                        key = lambda x: x[0]))
    print("%d %d %s:\n  %s" % (weight[node], cost[node], node,
        "\n  ".join("%s %s" % x for x in deps)))

# create networkx graph
G=nx.DiGraph()

for node in nodes:
    G.add_node(node)
for edge in edges:
    G.add_edge(*edge)

# draw graph
pos = nx.spring_layout(G)
#pos = nx.shell_layout(G)
nx.draw_networkx_labels(G, pos, {n:n for n in nodes})
node_sizes = [10 * cost[node] for node in nodes]
nx.draw(G, pos, node_size=node_sizes)

# show graph
plt.show()


