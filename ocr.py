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
            print("Neural network loaded from file.")

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
        print(f"Training with {len(training_data_array)} samples...")
        for data in training_data_array:
            try:
                # Ensure data is in the right format
                if isinstance(data, tuple):
                    data = {"y0": data[0], "label": int(data[1])}
                
                # Ensure y0 is a list
                if 'y0' not in data or not data['y0']:
                    print(f"Warning: Missing or empty 'y0' in training data")
                    continue
                    
                # Ensure label is an integer
                if 'label' not in data:
                    print(f"Warning: Missing 'label' in training data")
                    continue
                
                label = data['label']
                if isinstance(label, str):
                    try:
                        label = int(label)
                    except ValueError:
                        print(f"Warning: Label '{label}' is not a valid number")
                        continue
                
                # Ensure label is in range 0-9
                if not (0 <= label <= 9):
                    print(f"Warning: Label {label} outside valid range 0-9")
                    continue
                    
                # Ensure y0 is properly formatted
                y0_data = data['y0']
                if len(y0_data) != 400:
                    print(f"Warning: Input data length {len(y0_data)} != 400")
                    continue
                
                # Convert y0 to a proper matrix format - this is likely the issue
                y0_matrix = np.asmatrix([float(val) for val in y0_data])
                
                # Forward propagation
                y1 = np.dot(np.asmatrix(self.theta1), y0_matrix.T)
                sum1 = y1 + np.asmatrix(self.input_layer_bias)  # add bias
                y1 = self.sigmoid(sum1)

                y2 = np.dot(np.array(self.theta2), y1)
                y2 = y2 + np.array(self.hidden_layer_bias)
                y2 = self.sigmoid(y2)

                # Back propagation
                actual_values = [0] * 10
                actual_values[int(label)] = 1  # Use integer label as index
                output_errors = np.asmatrix(actual_values).T - np.asmatrix(y2)
                hidden_errors = np.multiply(np.dot(np.asmatrix(self.theta2).T, output_errors), self.sigmoid_prime(sum1))

                # Update weights
                self.theta1 += self.LEARNING_RATE * np.dot(hidden_errors, y0_matrix)
                self.theta2 += self.LEARNING_RATE * np.dot(output_errors, np.asmatrix(y1).T)
                self.hidden_layer_bias += self.LEARNING_RATE * output_errors
                self.input_layer_bias += self.LEARNING_RATE * hidden_errors
                
                print(f"Successfully trained on digit {label}")
            
            except Exception as e:
                print(f"Error training on sample: {str(e)}")
                import traceback
                traceback.print_exc()
                
        print("Training complete.")

    def predict_with_confidence(self, test):
        """Returns confidence scores for all digits and the predicted digit"""
        try:
            # Convert input to appropriate format if needed
            if isinstance(test, str):
                test = [int(c) for c in test]
                
            # Validate input data
            if len(test) != 400:
                print(f"Warning: Input data length {len(test)} != 400")
                return 9, {i: 0.0 for i in range(10)}  # Default to 9 with zero confidence
                
            # Convert to matrix format with proper float values
            test_matrix = np.asmatrix([float(val) for val in test])
            
            # Forward propagation
            y1 = np.dot(np.array(self.theta1), test_matrix.T)
            y1 = y1 + np.asmatrix(self.input_layer_bias)
            y1 = self.sigmoid(y1)

            y2 = np.dot(np.array(self.theta2), y1)
            y2 = np.add(y2, self.hidden_layer_bias)
            y2 = self.sigmoid(y2)

            # Get confidence scores for all digits
            results = y2.T.tolist()[0]
            confidence_scores = {i: float(results[i]) for i in range(len(results))}
            
            # Print confidence scores for debugging
            print("Confidence scores for each digit:")
            for digit, confidence in confidence_scores.items():
                print(f"Digit {digit}: {confidence:.6f}")
            
            # Return the digit with highest confidence
            predicted_digit = results.index(max(results))
            return predicted_digit, confidence_scores
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return 9, {i: 0.0 for i in range(10)}  # Default to 9 with zero confidence

    def predict(self, test):
        """Returns the predicted digit (for backward compatibility)"""
        predicted_digit, _ = self.predict_with_confidence(test)
        return predicted_digit
    
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
        print(f"Neural network saved to {OCRNeuralNetwork.NN_FILE_PATH}")

    def _load(self):
        if not self._use_file:
            return
        
        try:
            with open(OCRNeuralNetwork.NN_FILE_PATH) as nnFile:
                nn = json.load(nnFile)
            self.theta1 = [np.array(li) for li in nn['theta1']]
            self.theta2 = [np.array(li) for li in nn['theta2']]
            self.input_layer_bias = np.array(nn['input_layer_bias'])
            self.hidden_layer_bias = np.array(nn['hidden_layer_bias'])
        except Exception as e:
            print(f"Error loading neural network: {str(e)}")
            raise
