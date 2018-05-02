import random
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def rand():
    # Random weight between -0.2 and 0.2
    return random.random() * 0.4 - 0.2


class Neuron:
    def __init__(self, n_inputs):
        self.weights = [rand() for _ in range(n_inputs)]
        self.bias = rand()
        self.last_output = 0
        self.last_input = []
        self.error = 0
        self.errors = {}

    def process(self, inputs):
        self.last_input = inputs

        out = 0
        for i in range(len(inputs)):
            out += inputs[i] * self.weights[i]
        out += self.bias

        self.last_output = sigmoid(out)
        return self.last_output


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def process(self, inputs):
        return [self.neurons[i].process(inputs) for i in range(len(self.neurons))]


class Network:
    def __init__(self, training_iteration=500000, learning_rate=0.3, error_threshold=0.0001):
        self.layers = []
        self.training_iteration = training_iteration
        self.learning_rate = learning_rate
        self.error_threshold = error_threshold

    def add_layer(self, n_neurons=1, n_inputs=0):
        if n_inputs == 0:
            n_inputs = len(self.layers[-1].neurons)

        self.layers.append(Layer(n_inputs, n_neurons))

    def process(self, inputs):
        outputs = []
        for i in range(len(self.layers)):
            outputs = self.layers[i].process(inputs)
            inputs = outputs
        return outputs

    def train(self, examples):
        output_layer = self.layers[-1]

        for it in range(self.training_iteration):
            for e in range(0, len(examples)):
                example = examples[e]
                inputs = example[0]
                targets = example[1]
                outputs = self.process(inputs)

                for i in range(len(output_layer.neurons)):
                    neuron = output_layer.neurons[i]
                    neuron.error = targets[i] - outputs[i]
                    neuron.delta = neuron.last_output * (1 - neuron.last_output) * neuron.error

                    # Keep track of the error of each examples to determine when to stop training.
                    neuron.errors[e] = neuron.error

                self.back_propagation()

            # Compute the mean squared error for all examples.
            mse = self.get_mse()

            if it % 10000 == 0:
                print('iteration: {}, mse: {}'.format(it, mse))

            if mse < self.error_threshold:
                print('iteration: {}, mse: {}'.format(it, mse))
                return

    def back_propagation(self):
        for l in range(len(self.layers) - 2, -1, -1):
            for j in range(0, len(self.layers[l].neurons)):
                n_back = self.layers[l].neurons[j]
                n_back.error = 0
                for n_front in self.layers[l + 1].neurons:
                    n_back.error += n_front.weights[j] * n_front.delta
                n_back.delta = n_back.last_output * (1 - n_back.last_output) * n_back.error

                for i in range(0, len(self.layers[l + 1].neurons)):
                    n_front = self.layers[l + 1].neurons[i]
                    for w in range(len(n_front.weights)):
                        n_front.weights[w] += self.learning_rate * n_front.last_input[w] * n_front.delta
                    n_front.bias += self.learning_rate * n_front.delta

    def get_mse(self):
        output_layer = self.layers[-1]
        mse, cnt = 0, 0
        for n in output_layer.neurons:
            for e in n.errors:
                cnt += 1
                mse += n.errors[e] * n.errors[e]
        mse /= cnt
        return mse

