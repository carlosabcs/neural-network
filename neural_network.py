# from files_parser import parse_dataset_file
# from files_parser import parse_network_configuration
# from files_parser import parse_initial_weights
from files_parser import *
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})


class NeuralNetwork:
    def __init__(self, network_file = None, initial_weights_file = None, dataset_file = None):
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
            self.n_layers = 2 + len(self.hidden_layers_sizes)

        if initial_weights_file:
            self.weights, self.weights_without_bias = parse_initial_weights(initial_weights_file)

        if dataset_file:
            self.data, self.outputs = parse_dataset_file(dataset_file)


    def __calculate_error(self, predicted, target):
        # changed by Claudia
        first_mult = np.multiply((-1 * target), np.log(predicted))
        second_mult = np.multiply((1 - target), np.log(1 - predicted))

        return np.sum(first_mult - second_mult)


    def __calculate_gradient(self, activations, delta):
        len_activations = len(activations)
        activations_reshape = activations.reshape((len_activations, 1))
        deltas_reshape = delta.reshape(len(delta),1)
        delta_by_activations = np.multiply(
                                    deltas_reshape,
                                    np.transpose(activations_reshape))

        return delta_by_activations

    def __sigmoid_function(self, z):
        return 1 / (1 + np.exp(-1 * z))

    def __propagate_input(self, data):
        activations = [[] for i in range(self.n_layers)]

        # Append bias term
        activations[0] = np.insert(data, 0, 1)
        print('%sa%s: %s' % (' ' * 8, 1, data))
        print('')
        # Run layer by layer
        for k in range(1, self.n_layers):
            z = np.dot(self.weights[k - 1], activations[k - 1])
            a = self.__sigmoid_function(z)
            activations[k] = np.concatenate(([1], a), axis=0)

            if k == self.n_layers - 1:
                # For the last layer
                activations[k] = activations[k][1:]
                self.predictions.append(a)

            print('%sz%s: %s' % (' ' * 8, k + 1, z))
            print('%sa%s: %s' % (' ' * 8, k + 1, activations[k]))
            print('')

        return activations

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
            print((' ' * 8 ) + 'x: ', item)
            print((' ' * 8 ) + 'y: ', self.outputs[i])

        print('-------------------------------------------------------------')
        print('Calculando erro/custo J da rede')
        J = 0
        activations_by_example = []
        for i in range(len(self.data)):
            print(' ' * 4 + 'Procesando exemplo de treinamento', i + 1)
            print(' ' * 4 + 'Propagando entrada: ', self.data[i])
            activations = self.__propagate_input(self.data[i])

            # Show prediction
            predicted = self.predictions[-1]
            print('\n%sf(x): %s' % (' ' * 8, predicted))
            print(' ' * 4 + 'Saída predita para o exemplo %s: %s' % (i+1, predicted))
            print(' ' * 4 + 'Saída esperada para o exemplo %s: %s' % (i+1, self.outputs[i]))
            # Calculate error
            error = self.__calculate_error(predicted, self.outputs[i])
            print(' ' * 4 + 'J do exemplo %s: %.3f\n' % (i+1, error))
            J += error
            activations_by_example.append(activations.copy())

        # Total error with regularization
        J = J / len(self.data)
        S = np.sum([np.sum(layer_weights) for layer_weights in (self.weights_without_bias ** 2)])
        S *= (self.reg_factor / (2 * len(self.data)))
        print('J total do dataset (com regularizacao): %.5f\n\n' % (J + S))

        print('-------------------------------------------------------------')
        print('Rodando backpropagation')

        gradients_accumulated = [[] for i in range(self.n_layers)]
        gradients = [[] for i in range(self.n_layers)]
        for i in range(len(self.data)):
            print(' ' * 4 + 'Calculando gradientes com base no exemplo', i + 1)
            deltas = [[] for i in range(self.n_layers)]

            for k in range(self.n_layers, 1, -1):
                if k == self.n_layers:
                    # Output layer deltas
                    deltas[k - 1] = self.predictions[i] - self.outputs[i]
                else:
                    # Deltas for hidden layers
                    deltas[k - 1] = np.multiply(
                        np.multiply(
                            np.dot(np.transpose(self.weights[k - 1]), deltas[k]),
                            activations_by_example[i][k - 1]
                        ),
                        ( 1 - activations_by_example[i][k - 1] )
                    )
                    deltas[k - 1] = deltas[k - 1][1:]

                print(' ' * 8 + 'delta%s: %s' % (k, deltas[k - 1]))
                gradients[k - 2] = self.__calculate_gradient(activations_by_example[i][k - 2], deltas[k - 1])
                if gradients_accumulated[k - 2] == []:
                    gradients_accumulated[k - 2] = gradients[k - 2]
                else:
                    gradients_accumulated[k - 2] = gradients_accumulated[k - 2] + gradients[k - 2]

            for j in range(self.n_layers - 2, -1, -1):
                print('%sGradientes de Theta%d com base no exemplo %d:' % (' ' * 8, j + 1, i + 1))
                for row in gradients[j]:
                    print('%s%s' % (' ' * 12, row))
                print()

        print('%sDataset completo processado. Calculando gradientes regularizados' %(' ' * 4))
        for i in range(self.n_layers - 1):
            print('%sGradientes finais para Theta%d (com regularizacao):' %(' ' * 8, i + 1))
            # Add a column with zeros instead of the bias column
            new_weights = np.zeros(self.weights[i].shape)
            new_weights[:,1:] = self.weights_without_bias[i]

            P = self.reg_factor * new_weights
            gradients_accumulated[i] = (gradients_accumulated[i] + P) / len(self.data)
            for row in gradients_accumulated[i]:
                print('%s%s' % (' ' * 12, row))
            print()
