import os

file_map = {}
file_map["fear"] = {}
file_map["neutral"] = {}
file_map["fear"]["l"] = []
file_map["fear"]["r"] = []
file_map["neutral"]["l"] = []
file_map["neutral"]["r"] = []

for f in os.listdir("."):
    if ".bmp" not in f: continue

    fear = "fear" if "NWF" in f else "neutral"
    direction = "l" if "_l.bmp" in f else "r"
    file_map[fear][direction].append(f)

for cond in file_map.keys():
    for direction in file_map[cond].keys():
        for i,f in enumerate(file_map[cond][direction]):
            os.system("mv %s %s_%s_%d.bmp" % (f, cond, direction, i))
