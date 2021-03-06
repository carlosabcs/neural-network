from files_parser import *
np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})


class NeuralNetwork:
    def __init__(
        self,
        network_file,
        data_instance = None,
        initial_weights_file = None,
        dataset_file = None,
        target_attribute = None
    ):
        self.target_attribute = target_attribute
        self.predictions = []

        if data_instance is not None:
            self.target_attributes = [ # The same number as target attribute columns
                col for col in data_instance.index if col.startswith(self.target_attribute)
            ]
            self.reg_factor,\
            self.alpha,\
            self.batch_size,\
            self.hidden_layers_sizes = parse_network_configuration_for_dataset(network_file)
            self.output_layer_size = len(self.target_attributes)
            self.input_layer_size = data_instance.shape[0] - self.output_layer_size # Minus target attributes
            self.n_hidden_layers = len(self.hidden_layers_sizes)
            self.n_layers = 2 + len(self.hidden_layers_sizes)
        else:
            self.reg_factor,\
            self.input_layer_size,\
            self.output_layer_size,\
            self.hidden_layers_sizes = parse_network_configuration(network_file)
            self.n_hidden_layers = len(self.hidden_layers_sizes)
            self.n_layers = 2 + len(self.hidden_layers_sizes)

        if initial_weights_file:
            self.weights, self.weights_without_bias = parse_initial_weights(initial_weights_file)

        if dataset_file:
            self.data, self.outputs = parse_dataset_file(dataset_file)


    def __split_dataframe(self, data):
        inputs = data.drop(self.target_attributes, axis=1).values
        outputs = data[self.target_attributes].values
        return inputs, outputs


    def __indexOfGreatestValue(self, array):
        max_value = -1
        ind_value = -1
        for i, value in enumerate(array):
            if value > max_value:
                max_value = value
                ind_value = i
        return ind_value


    def fit(self, data):
        self.__initialize_random_weights() # Reset weights
        self.data, self.outputs = self.__split_dataframe(data)
        # TODO: reemplazar esto por un criterio de parada mejor definido
        for it in range(1000):
            error, hit_count, n_measures =  0.0, 0, 0
            
            gradients_accumulated = [[] for i in range(self.n_layers)]
            for batch_i in range(int(len(self.data) / self.batch_size)): # batch by batch
                start, end = self.batch_size * batch_i, self.batch_size * (batch_i + 1)
                self.predictions = []
                data_batch = self.data[start:end]
                outputs_batch = self.outputs[start:end]
                J, activations_by_example = self.__calculate_J(
                    data_batch,
                    outputs_batch,
                    self.weights_without_bias,
                    False
                )

                gradients = [[] for i in range(self.n_layers)]
                for i in range(len(data_batch)):
                    index_predicted = self.__indexOfGreatestValue(self.predictions[i])
                    index_expected = self.__indexOfGreatestValue(outputs_batch[i])
                    for idx in range(len(self.predictions[i])):
                        if idx == index_predicted:
                            self.predictions[i][idx] = 1
                        else:
                            self.predictions[i][idx] = 0
    
                    # Get current score:
                    error += np.sum((self.predictions[i] - outputs_batch[i]) ** 2)
                    # Increment hit count:
                    hit_count += int(index_predicted == index_expected)
                    n_measures += 1
                    deltas = [[] for j in range(self.n_layers)]
                    for k in range(self.n_layers, 1, -1):
                        if k == self.n_layers:
                            # Output layer deltas
                            deltas[k - 1] = self.predictions[i] - outputs_batch[i]
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

                        gradients[k - 2] = self.__calculate_gradient(activations_by_example[i][k - 2], deltas[k - 1])
                        if gradients_accumulated[k - 2] == []:
                            gradients_accumulated[k - 2] = gradients[k - 2]
                        else:
                            gradients_accumulated[k - 2] = gradients_accumulated[k - 2] + gradients[k - 2]

            for i in range(self.n_layers - 1):
                # Add a column with zeros instead of the bias column
                new_weights = np.zeros(self.weights[i].shape)
                new_weights[:,1:] = self.weights_without_bias[i]

                P = self.reg_factor * new_weights
                gradients_accumulated[i] = (gradients_accumulated[i] + P) / len(self.data)

            # Update weights
            gradients_accumulated = np.array(gradients_accumulated)
            self.weights = self.weights - (self.alpha * gradients_accumulated[:-1])
            self.weights_without_bias = self.weights.copy()
            for i in range(len(self.weights_without_bias)):
                self.weights_without_bias[i] = self.weights_without_bias[i][:,1:]

            if ((it + 1) % 50 == 0):
                print('Iteration %s, error: %.5f, accuracy: %.5f' % (it + 1, error / n_measures, hit_count / n_measures))


    def predict(self, data):
        self.data, self.outputs = self.__split_dataframe(data)
        hit_count = 0
        for idx, instance in enumerate(self.data):
            predicted = self.__propagate_input(instance, False)[-1]
            index_predicted = self.__indexOfGreatestValue(predicted)
            index_expected = self.__indexOfGreatestValue(self.outputs[idx])
            hit_count += int(index_predicted == index_expected)
        accuracy = hit_count / len(self.data)
        print('Test accuracy = %.5f\n' % (accuracy))
        return accuracy, hit_count


    def __initialize_random_weights(self):
        weights = []
        weights_without_bias = []
        # First layer -> hidden_0 * input_size
        weights.append(
            np.random.uniform(
                low=-1,
                high=1,
                size=(self.hidden_layers_sizes[0], self.input_layer_size + 1)
            )
        )
        weights_without_bias.append(
            weights[-1][:, 1:]
        )

        # Intermediate layers -> hidden_i+1 * hidden_i
        for i in range(0, len(self.hidden_layers_sizes) - 1):
            weights.append(
                np.random.uniform(
                    low=-1,
                    high=1,
                    size=(self.hidden_layers_sizes[i+1], self.hidden_layers_sizes[i] + 1)
                )
            )
            weights_without_bias.append(
                weights[-1][:, 1:]
            )

        #  Last layer -> output_size * self.hidden_layers_sizes[i]
        weights.append(
            np.random.uniform(
                low=-1,
                high=1,
                size=(self.output_layer_size, self.hidden_layers_sizes[-1] + 1)
            )
        )
        weights_without_bias.append(
            weights[-1][:, 1:]
        )

        self.weights = np.array(weights)
        self.weights_without_bias = np.array(weights_without_bias)


    def __calculate_error(self, predicted, target):
        # Changed by Claudia
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


    def __calculate_J(
        self,
        data,
        outputs,
        weights,
        log=True
    ):
        partial_J = 0
        activations_by_example = []
        for i in range(len(data)):
            activations = self.__propagate_input(data[i], log)
            # Show prediction
            predicted = self.predictions[-1]
            # Calculate error
            error = self.__calculate_error(predicted, outputs[i])
            if log:
                print(' ' * 4 + 'Procesando exemplo de treinamento', i + 1)
                print(' ' * 4 + 'Propagando entrada: ', data[i])
                print('\n%sf(x): %s' % (' ' * 8, predicted))
                print(' ' * 4 + 'Saída predita para o exemplo %s: %s' % (i+1, predicted))
                print(' ' * 4 + 'Saída esperada para o exemplo %s: %s' % (i+1, outputs[i]))
                print(' ' * 4 + 'J do exemplo %s: %.3f\n' % (i+1, error))
            partial_J += error
            activations_by_example.append(activations.copy())

        J = partial_J / len(data)
        S = np.sum([np.sum(layer_weights) for layer_weights in (weights ** 2)])
        S *= (self.reg_factor / (2 * len(data)))

        return J + S, activations_by_example


    def __propagate_input(self, data, log=True):
        activations = [[] for i in range(self.n_layers)]

        # Append bias term
        activations[0] = np.insert(data, 0, 1)
        if log:
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

            if log:
                print('%sz%s: %s' % (' ' * 8, k + 1, z))
                print('%sa%s: %s' % (' ' * 8, k + 1, activations[k]))
                print('')

        return activations

    def __calculate_J2(self, data, target):
        predicted = self.__propagate_input(data, False)[-1]
        error = self.__calculate_error(predicted, target)

        return error

    def __calculate_numerical_gradient(self, e, expected_gradients):
        print("--------------------------------------------")
        print("Rodando verificacao numerica de gradientes (epsilon=%.10f)" % e)
        accumulated_gradients = None

        for idx, data in enumerate(self.data):
            gradients_by_data = []
            for i in range(len(self.weights)):
                matrix = np.zeros(self.weights[i].shape)
                for j in range(len(self.weights[i])):
                    for k in range(len(self.weights[i][j])):
                        self.weights[i][j][k] = self.weights[i][j][k] + e
                        first_out = self.__calculate_J2(data, self.outputs[idx])
                        self.weights[i][j][k] = self.weights[i][j][k] - (2 * e)
                        second_out = self.__calculate_J2(data, self.outputs[idx])
                        self.weights[i][j][k] = self.weights[i][j][k] + e
                        #Calculate gradient
                        matrix[j][k]= (first_out - second_out)/(2 * e)
                gradients_by_data.append(matrix)
            gradients_by_data = np.array(gradients_by_data)

            if accumulated_gradients == None:
                accumulated_gradients = gradients_by_data.copy()
            else:
                accumulated_gradients = accumulated_gradients + gradients_by_data

        accumulated_gradients = np.array(accumulated_gradients)

        errors = []
        # Gradients with regularization
        for i in range(len(accumulated_gradients)):
            print('%sGradientes numerico de Theta%d:' %(' ' * 4, i + 1))
            # Add a column with zeros instead of the bias column
            new_weights = np.zeros(accumulated_gradients[i].shape)
            new_weights[:,1:] = self.weights_without_bias[i].copy()

            P = self.reg_factor * new_weights
            accumulated_gradients[i] = (accumulated_gradients[i] + P) / len(self.data)
            errors.append(np.sum((expected_gradients[i] - accumulated_gradients[i])**2))

            for row in accumulated_gradients[i]:
                print('%s%s' % (' ' * 8, row))
            print()

        print('Verificando corretude dos gradientes com base nos gradientes numericos:')
        print('%sError = np.sum(gradiente via backprop - gradiente numerico)**2' % (' ' * 4))
        for idx, err in enumerate(errors):
            print('%sErro entre gradiente via backprop e gradiente numerico para Theta%d: %0.20f' % (' ' * 4, idx + 1, err))

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
        # Total error with regularization
        J, activations_by_example = self.__calculate_J(self.data, self.outputs, self.weights_without_bias)
        print('J total do dataset (com regularizacao): %.5f\n\n' % J)

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

        # Run calculation of numerical gradients
        self.__calculate_numerical_gradient(0.0000010000, gradients_accumulated)



