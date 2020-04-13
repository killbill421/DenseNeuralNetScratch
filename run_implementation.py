import numpy as np

# THE ANN ARCHITECTURE 
# Layer 1 = 64 neurons 
# layer 2 = 128 neurons 
# layer 3 = 128 neurons
# Output layer = ? neurons 
# loss Fxn = ? 
# output layer will depend on the data and the loss function we choose 
# We will try to make the generalized architecture that is scalable 
# We will be choosing the class method instead of the functional approch

class gen_ANN:
    def __init__(self, X, Y, count_neuron, sizes_neuron):
        self.X = X
        self.Y = Y
        self.count_neuron = count_neuron
        self.sizes_neuron = sizes_neuron
        self.input_layer = 100
        # Generalized shape of activation and biases created
        self.activations = [np.random.randn(self.input_layer, self.sizes_neuron[i]) if i == 0 else np.random.randn(self.sizes_neuron[i], self.sizes_neuron[i - 1]) for i in range(len(self.sizes_neuron))]
        self.biases = [np.random.randn(size, 1) for size in self.sizes_neuron]
        # done
    def test1(self):
        return [n.shape for n in self.activations], [n.shape for n in self.biases]


# model = gen_ANN(3, [32, 64, 128])
# activations, biases = model.test1()
# print(activations)
# print(biases)
        
