import argparse
import numpy as np
import os
import json
import pandas as pd
from neural_network import NeuralNetwork
from cross_validator import CrossValidator


DATA_PATH = './data/'


def preprocess(df, types):
    # One hot encoding
    non_numeric_columns = []
    numeric_columns = []
    for col in types.keys():
        if df[col].dtype != 'object':
            numeric_columns.append(col)
            continue
        non_numeric_columns.append(col)
    df = pd.get_dummies(df, columns=non_numeric_columns)

    # Data normalization
    for col in numeric_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = df[col].apply(lambda x: ((x - min_val) / (max_val - min_val)))
    return df


def main():
    parser = argparse.ArgumentParser(description='Multilayer neural network parser')
    parser.add_argument('-d', '--dataset', help='The name (without extension) of the dataset', required=True)
    parser.add_argument('-n', '--network', help='The filename of the network configuration', required=False)
    args = parser.parse_args()

    try:
        with open(DATA_PATH + args.dataset + '.json', 'r') as filetypes:
            types = json.load(filetypes)
    except:
        print('Dataset types not found, automatic types will be used.')
        types = {}

    df = preprocess(
        pd.read_csv(DATA_PATH + args.dataset + '.tsv', sep='\t', dtype=types),
        types
    )
    network_file = open(args.network, 'r')
    cv = CrossValidator(
        NeuralNetwork(
            network_file=network_file,
            target_attribute='target',
            data_instance=df.iloc[0]
        )
    )
    cv.cross_validate(df, 10, 1)


if __name__ == "__main__":
    main()