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
        # self.reg_factor = 1
        # self.input_layer_size = 0
        # self.output_layer_size = 0
        # self.hidden_layers_sizes = []
        # self.data = []
        # self.outputs = []
        self.predictions = []
        if network_file:
            self.reg_factor,\
            self.input_layer_size,\
            self.output_layer_size,\
            self.hidden_layers_sizes = parse_network_configuration(network_file)
            self.n_layers = 1 + len(self.hidden_layers_sizes) + 1

        if initial_weights_file:
            self.weights, self.weights_without_bias = parse_initial_weights(initial_weights_file)

        if dataset_file:
            self.data, self.outputs = parse_dataset_file(dataset_file)


    def __calculate_error(self, predicted, target):
        return np.sum(
            np.multiply((-1 * target), np.log(predicted)) - np.multiply((1 - target), np.log(1 - predicted))
        )


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
        J = 0
        activations = [[] for i in range(self.n_layers)]
        for i in range(len(self.data)):
            print(' ' * 4 + 'Procesando exemplo de treinamento', i + 1)
            print(' ' * 4 + 'Propagando entrada: ', self.data[i])

            # Append bias term
            data = [np.insert(self.data[i], 0, 1)]
            activations[0] = data[0]
            print('%sa%s: %s' % (' ' * 8, 1, data[0]))
            print('')
            # Run layer by layer
            for k in range(1, self.n_layers - 1):
                z = np.dot(self.weights[k - 1], data[k - 1])
                a = 1 / (1 + np.exp(-1 * z))
                activations[k] = np.concatenate(([1], a), axis=0)
                data.append(activations[k])
                print('%sz%s: %s' % (' ' * 8, k + 1, z))
                print('%sa%s: %s' % (' ' * 8, k + 1, data[k]))
                print('')
            # Run last layer
            last_layer_idx = len(self.hidden_layers_sizes)
            z = np.dot(self.weights[last_layer_idx], data[last_layer_idx])
            a = 1 / (1 + np.exp(-1 * z))
            activations[self.n_layers - 1] = a
            self.predictions.append(a)
            print('%sz%s: %s' % (' ' * 8, last_layer_idx + 2, z))
            print('%sa%s: %s' % (' ' * 8, last_layer_idx + 2, a))
            print('\n%sf(x): %s' % (' ' * 8, a))
            print(' ' * 4 + 'Saída predita para o exemplo %s: %s' % (i+1, a))
            print(' ' * 4 + 'Saída esperada para o exemplo %s: %s' % (i+1, self.outputs[i]))
            # Calculate error
            error = self.__calculate_error(a, self.outputs[i])
            print(' ' * 4 + 'J do exemplo %s: %.3f\n' % (i+1, error))
            J += error

        # Total error with regularization
        J = J / len(self.data)
        S = np.sum([np.sum(layer_weights) for layer_weights in (self.weights_without_bias ** 2)])
        S *= (self.reg_factor / (2 * len(self.data)))
        print('J total do dataset (com regularizacao): %.5f\n\n' % (J + S))

        print('-------------------------------------------------------------')
        print('Rodando backpropagation')
        for i in range(len(self.data)):
            print(' ' * 4 + 'Calculando gradientes com base no exemplo', i + 1)
            deltas = [[] for i in range(self.n_layers)]
            # Output layer deltas
            deltas[self.n_layers - 1] = self.predictions[i] - self.outputs[i]
            print(' ' * 8 + 'delta%s: %s' % (self.n_layers, deltas[self.n_layers - 1]))
            for k in range(self.n_layers - 1, 1, -1):
                deltas[k - 1] = np.multiply(
                    np.multiply(
                        np.dot(np.transpose(self.weights[k - 1]), deltas[k]),
                        activations[k - 1]
                    ),
                    ( 1 - activations[k - 1] )
                )
                deltas[k - 1] = deltas[k - 1][1:]
                print(' ' * 8 + 'delta%s: %s' % (k, deltas[k - 1]))

