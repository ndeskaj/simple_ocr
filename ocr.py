import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from numpy import matrix
from numpy import pow
from collections import namedtuple
import math
import random
import os
import json

class OCRNeuralNetwork:
    LEARNING_RATE = 0.1
    WIDTH_IN_PIXELS = 20
    NN_FILE_PATH = 'nn.json'

    def __init__(self, num_hidden_nodes, data_matrix, data_labels, 
                 training_indices, use_file=True):
        self.sigmoid = np.vectorize(self._sigmoid_scalar)
        self.sigmoid_prime = np.vectorize(self._sigmoid_prime_scalar)
        self.data_matrix = data_matrix
        self.data_labels = data_labels
        self._use_file = use_file

        if (not os.path.isfile(OCRNeuralNetwork.NN_FILE_PATH) or not use_file):
            # First step: initialize nodes to small weights
            self.theta1 = self._rand_initialize_weights(400, num_hidden_nodes)
            self.theta2 = self._rand_initialize_weights(num_hidden_nodes, 10)
            self.input_layer_bias = self._rand_initialize_weights(1, num_hidden_nodes)
            self.hidden_layer_bias = self._rand_initialize_weights(1, 10)

            TrainData = namedtuple('TrainData', ['y0', 'label'])
            self.train([TrainData(matrix(data_matrix[i]), data_labels[i]) 
                        for i in training_indices])
        else:
            self._load()

    # randomize the weights for each layer in the neural network
    def _rand_initialize_weights(self, size_in, size_out):
        return [((x * 0.12) - 0.06) for x in np.random.rand(size_out, size_in)]
    

    # sigmoid activation function
    def _sigmoid_scalar(self, z):
        return 1 / (1 + math.e ** -z)
    
    # derivative of sigmoid function
    def _sigmoid_prime_scalar(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def _draw(self, sample):
        pixelArray = [sample[j:j+self.WIDTH_IN_PIXELS]for j in range(0, len(sample), self.WIDTH_IN_PIXELS)]
        plt.imshow(zip(*pixelArray), cmap = cm.Greys, interpolation='nearest')
        plt.show()

    def train(self, training_data_array):
        for data in training_data_array:
            if isinstance(data, tuple):
                data = {"y0": data[0], "label": int(data[1])}  # Ensure label is an integer

            # Second step: forward propagation
            y1 = np.dot(np.asmatrix(self.theta1), np.asmatrix(data["y0"]).T)
            sum1 = y1 + np.asmatrix(self.input_layer_bias)  # add bias
            y1 = self.sigmoid(sum1)

            y2 = np.dot(np.array(self.theta2), y1)
            y2 = y2 + np.array(self.hidden_layer_bias)
            y2 = self.sigmoid(y2)

            # Third step: back propagation
            actual_values = [0] * 10
            actual_values[data['label']] = 1  # Use integer label as index
            output_errors = np.asmatrix(actual_values).T - np.asmatrix(y2)
            hidden_errors = np.multiply(np.dot(np.asmatrix(self.theta2).T, output_errors), self.sigmoid_prime(sum1))

            # Fourth step: update weights
            self.theta1 += self.LEARNING_RATE * np.dot(hidden_errors, np.asmatrix(data['y0']))
            self.theta2 += self.LEARNING_RATE * np.dot(output_errors, np.asmatrix(y1).T)
            self.hidden_layer_bias += self.LEARNING_RATE * output_errors
            self.input_layer_bias += self.LEARNING_RATE * hidden_errors

    def predict(self, test):
        y1 = np.dot(np.array(self.theta1), np.asmatrix(test).T)
        y1 = y1 + np.asmatrix(self.input_layer_bias)
        y1 = self.sigmoid(y1)

        y2 = np.dot(np.array(self.theta2), y1)
        y2 = np.add(y2, self.hidden_layer_bias)
        y2 = self.sigmoid(y2)

        results = y2.T.tolist()[0]
        return results.index(max(results))
    
    def save(self):
        if not self._use_file:
            return
        
        json_neural_network = {
            "theta1":[np_mat.tolist()[0] for np_mat in self.theta1],
            "theta2":[np_mat.tolist()[0] for np_mat in self.theta2],
            "input_layer_bias":self.input_layer_bias[0].tolist()[0],
            "hidden_layer_bias":self.hidden_layer_bias[0].tolist()[0]
        }
        with open(OCRNeuralNetwork.NN_FILE_PATH, 'w') as nnFile:
            json.dump(json_neural_network, nnFile)

    def _load(self):
        if not self._use_file:
            return
        
        with open(OCRNeuralNetwork.NN_FILE_PATH) as nnFile:
            nn = json.load(nnFile)
        self.theta1 = [np.array(li) for li in nn['theta1']]
        self.theta2 = [np.array(li) for li in nn['theta2']]
        self.input_layer_bias = np.array(nn['input_layer_bias'])
        self.hidden_layer_bias = np.array(nn['hidden_layer_bias'])
