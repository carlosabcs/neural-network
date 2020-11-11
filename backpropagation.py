import argparse
import numpy as np
import os
from neural_network import NeuralNetwork


def run_neural_network(network_file, initial_weights_file, dataset_file):
    nn = NeuralNetwork(None, network_file, initial_weights_file, dataset_file)
    nn.test_backpropagation()


def main():
    parser = argparse.ArgumentParser(description='Multilayer neural network parser')
    parser.add_argument('-n', '--network', help='The filename of the network configuration',
                        default= 'network.txt' ,required=False)
    parser.add_argument('-i', '--initial_weights', help='The filename of the initial weights configuration',
                        default='initial_weights.txt', required=False)
    parser.add_argument('-d', '--dataset', help='The filename of the dataset',
                        default='dataset.txt', required=False)

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
