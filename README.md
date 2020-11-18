# Multilayer Neural Network

*This implementation is part of the course CMP263 Machine Learning, from Instituto de Informática (INF) - UFRGS*

## Installation
This project requires two libraries which can be installed by running: `pip install pandas numpy`

## Backpropagation and gradient numerical check
This project has an special script to be run for testing the backpropagation algorithm implementation and a numerical check for the gradient.

The script receives a set of arguments:
- `--network`: the file with the network configuration. That file has the following structure:
  ```python3
  regularization_factor (float)
  input_layer_size (int)
  hidden_layer_0_size (int)
  hidden_layer_1_size (int)
  ...
  hidden_layer_n_size (int)
  output_layer_size (int)
  ```
  An example in this project is the file `network.txt`.
- `--initial_weights`: the file with the initial weights to be used in the neural network trainning. That file has the structure:
  ```python3
  layer_0_neuron_0_0, layer_0_neuron_0_1, ..., layer_0_neuron_0_A; layer_0_neuron_1_0, layer_0_neuron_1_0, ..., layer_0_neuron_1_B
  layer_1_neuron_0_0, layer_1_neuron_0_1, ..., layer_1_neuron_0_C; layer_1_neuron_1_0, layer_1_neuron_1_0, ..., layer_1_neuron_1_D
  ...
  layer_n_neuron_0_0, layer_n_neuron_0_1, ..., layer_n_neuron_0_E; layer_n_neuron_1_0, layer_n_neuron_1_0, ..., layer_n_neuron_1_F
  ```
  Two examples of this file could be found in: `exemplo_backprop_rede1.txt` and `exemplo_backprop_rede2.txt`.
- `--dataset`: the file with the dataset to be used for network trainning. The file should have the following structure:
  ```python3
  ex_0_attr_0, ex_0_attr_1, ..., ex_0_attr_N; ex_0_output_0, ex_0_output_1, ..., ex_0_output_M
  ex_1_attr_0, ex_1_attr_1, ..., ex_1_attr_N; ex_1_output_0, ex_1_output_1, ..., ex_1_output_M
  ...
  ex_Z_attr_0, ex_Z_attr_1, ..., ex_Z_attr_N; ex_Z_output_0, ex_Z_output_1, ..., ex_Z_output_M
  ```

## Running the Multilayer Neural Network with a dataset

The script `main.py` trains and tests the multilayer neural network algorithm using a certain dataset. It receives the following parameters:
- `-d`: The name of the dataset file, in a `.tsv` format. The file should be in the `./data` folder.
- `-n`: The name of the network configuration file. This configuration is slightly different than the previous one, its structure is this:
  ```python3
  regularization_factor (float)
  alpha (float)
  batch_size (int)
  hidden_layer_0_size (int)
  hidden_layer_1_size (int)
  ...
  hidden_layer_n_size (int)
  ```
  The file `network_v2.txt` is a example of this type of configuration.

### Important details
- The algorithm uses k-fold (k = 10) for trainning and testing the network accuracy.
- The initial weights are randomnly loaded on each fold iteration, those weights could have a value in the range of -1.0 and 1.0.
- The input and output layers' sizes are automatically calculated from the dataset.

# Examples
- Running the script for backpropagation and gradient numerical check.

  `python backpropagation.py --network network.txt --initial_weights initial_weights.txt --dataset dataset.txtataset.txt`

  The output should be something like:
  ```bash
  Parâmetro de regularizacao lambda: 0.25

  Inicializando rede com a seguinte estrutura de neurônios por camadas: [2, 4, 3, 2]

  Theta1 inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):
      [0.42000 0.15000 0.40000]
      [0.72000 0.10000 0.54000]
      [0.01000 0.19000 0.42000]
      [0.30000 0.35000 0.68000]

  Theta2 inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):
      [0.21000 0.67000 0.14000 0.96000 0.87000]
      [0.87000 0.42000 0.20000 0.32000 0.89000]
      [0.03000 0.56000 0.80000 0.69000 0.09000]

  Theta3 inicial (pesos de cada neuronio, incluindo bias, armazenados nas linhas):
      [0.04000 0.87000 0.42000 0.53000]
      [0.17000 0.10000 0.95000 0.69000]

  Conjunto de treinamento:
      Exemplo  1
          x:  [0.32000 0.68000]
          y:  [0.75000 0.98000]
      Exemplo  2
          x:  [0.83000 0.02000]
          y:  [0.75000 0.28000]
  -------------------------------------------------------------
  Calculando erro/custo J da rede
          a1: [0.32000 0.68000]

          z2: [0.74000 1.11920 0.35640 0.87440]
          a2: [1.00000 0.67700 0.75384 0.58817 0.70566]

          z3: [1.94769 2.12136 1.48154]
          a3: [1.00000 0.87519 0.89296 0.81480]

          z4: [1.60831 1.66805]
          a4: [0.83318 0.84132]

      Procesando exemplo de treinamento 1
      Propagando entrada:  [0.32000 0.68000]

          f(x): [0.83318 0.84132]
      Saída predita para o exemplo 1: [0.83318 0.84132]
      Saída esperada para o exemplo 1: [0.75000 0.98000]
      J do exemplo 1: 0.791

          a1: [0.83000 0.02000]

          z2: [0.55250 0.81380 0.17610 0.60410]
          a2: [1.00000 0.63472 0.69292 0.54391 0.64659]

          z3: [1.81696 2.02468 1.37327]
          a3: [1.00000 0.86020 0.88336 0.79791]

          z4: [1.58228 1.64577]
          a4: [0.82953 0.83832]

      Procesando exemplo de treinamento 2
      Propagando entrada:  [0.83000 0.02000]

          f(x): [0.82953 0.83832]
      Saída predita para o exemplo 2: [0.82953 0.83832]
      Saída esperada para o exemplo 2: [0.75000 0.28000]
      J do exemplo 2: 1.944

  J total do dataset (com regularizacao): 1.90351


  -------------------------------------------------------------
  Rodando backpropagation
      Calculando gradientes com base no exemplo 1
          delta4: [0.08318 -0.13868]
          delta3: [0.00639 -0.00925 -0.00779]
          delta2: [-0.00087 -0.00133 -0.00053 -0.00070]
          Gradientes de Theta3 com base no exemplo 1:
              [0.08318 0.07280 0.07427 0.06777]
              [-0.13868 -0.12138 -0.12384 -0.11300]

          Gradientes de Theta2 com base no exemplo 1:
              [0.00639 0.00433 0.00482 0.00376 0.00451]
              [-0.00925 -0.00626 -0.00698 -0.00544 -0.00653]
              [-0.00779 -0.00527 -0.00587 -0.00458 -0.00550]

          Gradientes de Theta1 com base no exemplo 1:
              [-0.00087 -0.00028 -0.00059]
              [-0.00133 -0.00043 -0.00091]
              [-0.00053 -0.00017 -0.00036]
              [-0.00070 -0.00022 -0.00048]

      Calculando gradientes com base no exemplo 2
          delta4: [0.07953 0.55832]
          delta3: [0.01503 0.05809 0.06892]
          delta2: [0.01694 0.01465 0.01999 0.01622]
          Gradientes de Theta3 com base no exemplo 2:
              [0.07953 0.06841 0.07025 0.06346]
              [0.55832 0.48027 0.49320 0.44549]

          Gradientes de Theta2 com base no exemplo 2:
              [0.01503 0.00954 0.01042 0.00818 0.00972]
              [0.05809 0.03687 0.04025 0.03160 0.03756]
              [0.06892 0.04374 0.04775 0.03748 0.04456]

          Gradientes de Theta1 com base no exemplo 2:
              [0.01694 0.01406 0.00034]
              [0.01465 0.01216 0.00029]
              [0.01999 0.01659 0.00040]
              [0.01622 0.01346 0.00032]

      Dataset completo processado. Calculando gradientes regularizados
          Gradientes finais para Theta1 (com regularizacao):
              [0.00804 0.02564 0.04987]
              [0.00666 0.01837 0.06719]
              [0.00973 0.03196 0.05252]
              [0.00776 0.05037 0.08492]

          Gradientes finais para Theta2 (com regularizacao):
              [0.01071 0.09068 0.02512 0.12597 0.11586]
              [0.02442 0.06780 0.04164 0.05308 0.12677]
              [0.03056 0.08924 0.12094 0.10270 0.03078]

          Gradientes finais para Theta3 (com regularizacao):
              [0.08135 0.17935 0.12476 0.13186]
              [0.20982 0.19195 0.30343 0.25249]

  --------------------------------------------
  Rodando verificacao numerica de gradientes (epsilon=0.0000010000)
      Gradientes numerico de Theta1:
          [0.00804 0.02564 0.04987]
          [0.00666 0.01837 0.06719]
          [0.00973 0.03196 0.05252]
          [0.00776 0.05037 0.08492]

      Gradientes numerico de Theta2:
          [0.01071 0.09068 0.02512 0.12597 0.11586]
          [0.02442 0.06780 0.04164 0.05308 0.12677]
          [0.03056 0.08924 0.12094 0.10270 0.03078]

      Gradientes numerico de Theta3:
          [0.08135 0.17935 0.12476 0.13186]
          [0.20982 0.19195 0.30343 0.25249]
  ```

- Running the Multilayer Neural Network with a dataset

  `python main.py -d house-votes-84 -n network_v2.txt`

# Observations

- For some unknown reason, `numpy` is blocking us for using the size 2 for the first hidden layer on the neural network. So please avoid doing tests with a configuration that uses that size for the first hidden layer.
