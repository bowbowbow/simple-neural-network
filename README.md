# Simple Neural Network

A simple neural network implementation for AND, OR, and XOR.

- This `neural_network.py` with no more than 120 lines will help you understand how back propagation is used in neural networks.

## Usage

```python
from neural_network import Network

# When learning XOR operations

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

output = network.process([0, 1])
print('0 XOR 1 = {}'.format(output))
```


## Demo

- `python AND.py`
```
[output example]

iteration: 0, mse: 0.279369899398
iteration: 10000, mse: 0.140416196796
...
iteration: 490000, mse: 0.00407935729835

0 AND 0 = [0.00044918832916990373]
0 AND 1 = [0.05429299453341384]
1 AND 0 = [0.07005402918739255]
1 AND 1 = [0.9107181648130748]
```

- `python OR.py`
```
[output example]

iiteration: 0, mse: 0.287796555809
iteration: 10000, mse: 0.130495897628
...
iteration: 490000, mse: 0.00311697735165

0 OR 0 = [0.07836534022993959]
0 OR 1 = [0.9405820787998327]
1 OR 0 = [0.9505304498025197]
1 OR 1 = [0.9996987369500553]
```

- `python XOR.py`
```
[output example]

iteration: 0, mse: 0.287343442863
iteration: 10000, mse: 0.290867941968
...
iteration: 140368, mse: 9.99975334554e-05

0 XOR 0 = [0.008195213850176481]
0 XOR 1 = [0.9943253450728344]
1 XOR 0 = [0.9872772168051346]
1 XOR 1 = [0.011630962083775283]

```

Sometimes, demo scripts fails to learn operations. Because gradient descent method for backprogation can falls into the local optimal value.
