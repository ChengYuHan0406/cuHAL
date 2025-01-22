import pandas as pd
import json
from argparse import ArgumentParser

# Usage: python inference.py [model path] [colname path] [top]

parser = ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("colname_path")
parser.add_argument("top")
args = parser.parse_args()

colnames = list(pd.read_csv(args.colname_path, header=None)[0])

name_map = {}
for i in range(len(colnames)):
    name_map[i] = colnames[i]

filename = f"{args.model_path}"

with open(filename, 'r') as file:
    model = json.load(file)

interact = model['interaction']
thres = model['thres']
weight = model['weight']

model_size = len(weight)

summary = {}
for c in range(model_size):
    cur_interact = interact[c]
    cur_thres = thres[c]
    len_interact = len(cur_interact)
    key = ""
    for i in range(len_interact):
        key += f"I[{name_map[cur_interact[i]]} >= {cur_thres[i]}]"
        if (i < len_interact - 1):
            key += " * "

    summary[key] = weight[c]

summary = {k: v for k, v in sorted(summary.items(), key=lambda item: abs(item[1]), reverse=True)}
for i, key in enumerate(summary):
    if (i >= int(args.top)):
        break
    print(f"{key}: {summary[key]}")
