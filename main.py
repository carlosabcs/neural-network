import argparse
import numpy as np
import os
from neural_network import NeuralNetwork


def run_neural_network(
    network_file,
    initial_weights_file,
    dataset_file
):
    nn = NeuralNetwork(network_file, initial_weights_file, dataset_file)
    nn.test_backpropagation()


def main():
    parser = argparse.ArgumentParser(description='Multilayer neural network parser')
    parser.add_argument('--network', help='The filename of the network configuration', required=True)
    parser.add_argument('--initial_weights', help='The filename of the initial weights configuration', required=True)
    parser.add_argument('--dataset', help='The filename of the dataset', required=True)

    args = parser.parse_args()

    if not os.path.isfile(args.network):
        print('The network configuration file was not found.')
        return
    network_file = open(args.network, 'r')

    if not os.path.isfile(args.initial_weights):
        print('The initial weights configuration file was not found.')
        return
    initial_weights_file = open(args.initial_weights, 'r')

    if not os.path.isfile(args.dataset):
        print('The dataset file was not found.')
        return
    dataset_file = open(args.dataset, 'r')
    run_neural_network(network_file, initial_weights_file, dataset_file)


if __name__ == "__main__":
    main()
