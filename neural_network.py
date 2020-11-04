from files_parser import parse_dataset_file
from files_parser import parse_network_configuration
from files_parser import parse_initial_weights


class NeuralNetwork:
    def __init__(
        self,
        network_file = None,
        initial_weights_file = None,
        dataset_file = None
    ):
        # TODO: Use random when config files don't exist
        self.reg_factor = 1
        self.input_layer_size = 0
        self.output_layer_size = 0
        self.hidden_layers_sizes = []
        self.data = []
        self.outputs = []

        if network_file:
            self.reg_factor,\
            self.input_layer_size,\
            self.output_layer_size,\
            self.hidden_layers_sizes = parse_network_configuration(network_file)

        if initial_weights_file:
            self.weights = parse_initial_weights(initial_weights_file)

        if dataset_file:
            self.data, self.outputs = parse_dataset_file(dataset_file)
