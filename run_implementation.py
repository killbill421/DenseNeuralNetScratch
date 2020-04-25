import numpy as np
import pandas as pd
from tensorflow.keras.activations import sigmoid
// niggers iffy aah
# THE ANN ARCHITECTURE 
# Layer 1 = 64 neurons 
# layer 2 = 128 neurons 
# layer 3 = 128 neurons
# Output layer = ? neurons 
# loss Fxn = ? 
# output layer will depend on the data and the loss function we choose 
# We will try to make the generalized architecture that is scalable 
# We will be choosing the class method instead of the functional approch
// bitches got da stiffy ahh
class gen_ANN:
    def __init__(self, X, Y, sizes_neuron):
        self.X = X
        self.Y = Y
        self.sizes_neuron = sizes_neuron
        self.input_layer = X.shape[0]
        # Generalized shape of activation and biases created
        self.weights = [np.random.randn(self.sizes_neuron[i],self.input_layer) if i == 0 else np.random.randn(self.sizes_neuron[i], self.sizes_neuron[i - 1]) for i in range(len(self.sizes_neuron))]
        self.biases = [np.random.randn(size, 1) for size in self.sizes_neuron]

    def forward_pass(self):
        self.list_activations = []
        self.z = []
        self.z.append(np.dot(self.weights[0], self.X) + self.biases[0])
        print(self.z[0].shape)
        self.list_activations.append(sigmoid(self.z[0]))
        for i in range(1, len(self.weights)):
            z_out = np.dot(self.weights[i], self.list_activations[-1]) + self.biases[i]
            self.z.append(z_out)
            a_out = sigmoid(z_out)
            self.list_activations.append(a_out)
        return [n.shape for n in self.z], [n.shape for n in self.list_activations]


    def test1(self):
        return [n.shape for n in self.weights], [n.shape for n in self.biases]



#data = pd.read_csv("train.csv")
#features = [col for col in data.columns if col!= "label"]
#X = data["label"][:3]
#Y = data["label"][:3]
#model = gen_ANN(X, Y, [4, 1])
#print(model.test1())
#print(model.forward_pass())

        // gummo 
