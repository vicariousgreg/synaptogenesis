import io

filename = "iris.csv"

labels = []
for line in open(filename).readlines():
    label = line.strip().split(',')[-1]
    if label not in labels:
        labels.append(label)

input_file = open("iris_input.csv", "w+")
output_file = open("iris_output.csv", "w+")

max_val = 0
min_val = 0

for line in open(filename).readlines():
    input_file.write(",".join(line.strip().split(',')[:-1]) + "\n")

    output_line = ["0"] * len(labels)
    output_line[labels.index(line.strip().split(',')[-1])] = "1"
    output_file.write(",".join(output_line) + "\n")

input_file.close()
output_file.close()
