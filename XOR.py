from neural_network import Network

network = Network(training_iteration=500000, learning_rate=0.3, error_threshold=0.0001)
network.add_layer(5, 2)
network.add_layer(4)
network.add_layer(1)

network.train([
    [[0, 0], [0]],
    [[0, 1], [1]],
    [[1, 0], [1]],
    [[1, 1], [0]],
])

output = network.process([0, 0])
print('0 XOR 0 = {}'.format(output))

output = network.process([0, 1])
print('0 XOR 1 = {}'.format(output))

output = network.process([1, 0])
print('1 XOR 0 = {}'.format(output))

output = network.process([1, 1])
print('1 XOR 1 = {}'.format(output))
