import numpy as np
import pandas as pd

def optimizer():
    return


def activation_function(input, activation = "sigmoid"):
    if activation == "sigmoid":
        return 
    elif activation == "relu":
        return


class node():
    def __init__(self, in_dim, hidden_dim):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.W = np.random.rand(len(self.in_dim), self.hidden_dim)
        self.b = np.random.rand(self.hidden_dim)
    
    def forward(self, input):
        output = np.matmul(input, W) + b
        output = activation_function(output, activation="sigmoid")

        return output

    def backward(self, cost):
        return

class Model():
    def __init__(self, h_dims):
        self.h_dim = h_dims
    
    def forward(self, input):
        layers = []
        for i in range(len(self.h_dims)):
            for j in range(len(self.h_dims[i])):
                layers.append(node(input = input, hidden_dim = self.h_dims[i][j]))

        