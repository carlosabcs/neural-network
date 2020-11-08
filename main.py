import argparse
import numpy as np
import os
import json
import pandas as pd
from neural_network import NeuralNetwork


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
    datasets = [
        './data/house-votes-84',
        './data/wine-recognition'
    ]

    for dataset in datasets:
        try:
            with open(dataset + '.json', 'r') as filetypes:
                types = json.load(filetypes)
        except:
            print('Dataset types not found, automatic types will be used.')
            types = {}
        df = pd.read_csv(dataset + '.tsv', sep='\t', dtype=types)
        print(df)
        df = preprocess(df, types)
        print(df)


if __name__ == "__main__":
    main()