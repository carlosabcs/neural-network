# from files_parser import parse_dataset_file
# from files_parser import parse_network_configuration
# from files_parser import parse_initial_weights
from files_parser import *
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})


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


    def test_backpropagation(self):
        print('Parâmetro de regularizacao lambda: %s\n' % self.reg_factor)
        print(
            'Inicializando rede com a seguinte estrutura de neurônios por camadas: %s\n' %
            ([ self.input_layer_size ] + self.hidden_layers_sizes + [ self.output_layer_size ])
        )

        for i, layer_weights in enumerate(self.weights):
            print('Theta%s inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):' % (i + 1))
            for row in layer_weights:
                print('%s%s' % (' ' * 4, row))
            print('')

        print('Conjunto de treinamento:')
        for i, item in enumerate(self.data):
            print((' ' * 4 ) + 'Exemplo ', i + 1)
            print((' ' * 8) + 'x: ', item)
            print((' ' * 8 ) + 'y: ', self.outputs[i])

        print('-------------------------------------------------------------')
        print('Calculando erro/custo J da rede')
        for i in range(len(self.data)):
            print(' ' * 4 + 'Procesando exemplo de treinamento', i + 1)
            print(' ' * 4 + 'Propagando entrada: ', self.data[i])

            # Append bias term
            data = [np.insert(self.data[i], 0, 1)]
            print('%sa%s: %s' % (' ' * 8, 1, data[0]))
            print('')
            # Run layer by layer
            for k in range(1, 1 + len(self.hidden_layers_sizes)):
                z = np.dot(self.weights[k - 1], data[k - 1])
                a = 1 / (1 + np.exp(-1 * z))
                data.append(np.concatenate(([1], a), axis=0))
                print('%sz%s: %s' % (' ' * 8, k + 1, z))
                print('%sa%s: %s' % (' ' * 8, k + 1, data[k]))
                print('')
            # Run last layer
            last_layer_idx = len(self.hidden_layers_sizes)
            z = np.dot(self.weights[last_layer_idx], data[last_layer_idx])
            a = 1 / (1 + np.exp(-1 * z))
            print('%sz%s: %s' % (' ' * 8, last_layer_idx + 2, z))
            print('%sa%s: %s' % (' ' * 8, last_layer_idx + 2, a))
            print('\n%sf(x): %s' % (' ' * 8, a))
            print(' ' * 4 + 'Saída predita para o exemplo %s: %s' % (i+1, a))
            print(' ' * 4 + 'Saída esperada para o exemplo %s: %s' % (i+1, self.outputs[i]))
            # TODO: Calcular J
            print('')

