import numpy as np

#define input and output for training
input_train = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
output_train = np.array([[0, 1, 1, 0]]).T


class NN():
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
        
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        
    def __sigmoid(self,x):
        return 1/(1-np.exp(-x))
    
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

   

np.random.seed(1)   #This will generate a particular random number when random.random() is used

synp_input = 2 * np.random.random((3, 4)) - 1
synp_output = 2 * np.random.random((4, 1)) - 1
