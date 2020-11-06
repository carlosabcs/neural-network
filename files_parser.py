import numpy as np


def parse_network_configuration(network_file):
    lines = network_file.readlines()
    reg_factor = float(lines[0])
    input_layer_size = int(lines[1])
    output_layer_size = int(lines[-1])
    try:
        hidden_layers_sizes = [int(line) for line in lines[2:-1]]
    except:
        hidden_layers_sizes = []
    return reg_factor, input_layer_size, output_layer_size, hidden_layers_sizes


def parse_initial_weights(initial_weights_file):
    lines = initial_weights_file.readlines()
    layers_weights = []
    layers_weights_without_bias = []
    for line in lines:
        layer_weights = []
        layer_weights_without_bias = []
        for neurons_weights in line.split('; '):
            neuron_weights = []
            weights = neurons_weights.split(', ')
            for weight in weights:
                neuron_weights.append(float(weight))
            layer_weights.append(np.array(neuron_weights))
            layer_weights_without_bias.append(np.array(neuron_weights[1:]))
        layers_weights.append(np.array(layer_weights))
        layers_weights_without_bias.append(np.array(layer_weights_without_bias))
    return np.array(layers_weights), np.array(layers_weights_without_bias)


def parse_dataset_file(dataset_file):
    lines = dataset_file.readlines()
    data = []
    output = []
    for line in lines:
        input_data, output_data = line.split('; ')

        instance = []
        for input_element in input_data.split(', '):
            instance.append(float(input_element))
        data.append(np.array(instance))

        instance = []
        for output_element in output_data.split(', '):
            instance.append(float(output_element))
        output.append(np.array(instance))

    return np.array(data), np.array(output)
