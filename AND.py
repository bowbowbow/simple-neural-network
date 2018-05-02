from neural_network import Network

network = Network(training_iteration=500000, learning_rate=0.3, error_threshold=0.001)
network.add_layer(3, 2)
network.add_layer(1)

network.train([
    [[0, 0], [0]],
    [[0, 1], [0]],
    [[1, 0], [0]],
    [[1, 1], [1]],
])

output = network.process([0, 0])
print('0 AND 0 = {}'.format(output))

output = network.process([0, 1])
print('0 AND 1 = {}'.format(output))

output = network.process([1, 0])
print('1 AND 0 = {}'.format(output))

output = network.process([1, 1])
print('1 AND 1 = {}'.format(output))
