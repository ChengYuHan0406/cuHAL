import numpy as np
import pandas as pd
import json
from argparse import ArgumentParser

# Usage: python inference.py [model path] [input path] [output path]

def inference(model, X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    model_size = len(model['weight'])

    sample_size = X.shape[0]
    res = model['bias'] * np.ones(sample_size)

    for i in range(model_size):
        cur_weight = model['weight'][i]
        cur_interact = model['interaction'][i]
        cur_thres = model['thres'][i]

        len_interact = len(cur_interact)

        basis = np.ones(sample_size).astype(bool)
        for j in range(len_interact):
            basis = basis & (X[:, cur_interact[j]] >= cur_thres[j])
        res += cur_weight * basis

    return res

parser = ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("input_path")
parser.add_argument("output_path")
args = parser.parse_args()

filename = f"{args.model_path}"

with open(filename, 'r') as file:
    model = json.load(file)

X = pd.read_csv(args.input_path, header=None)
y_hat = inference(model, np.array(X))
pd.DataFrame(y_hat).to_csv(args.output_path, index=False, header=False)
