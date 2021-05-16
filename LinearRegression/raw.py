import argparse
import numpy as np
import pandas as pd

# define the model
class Model():
    def __init__(self, w):
       self.w = w
    
    def forward(self, input):
        return self.w*input 

# cost
def cost(y_pred, y_true):
    cost = (y_pred - y_true)**2 / len(y_true)
    return sum(cost)

# gradient
def gradient(x, y_pred, y_true):
    grad = (2*(y_pred - y_true)*x) / len(y_true)
    return sum(grad)
    
if __name__ == "__main__":
    # setting arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input_path", default="train_data.csv") # it must be csv file
    parser.add_argument("initial_weight", default=4)
    parser.add_argument("learning_rate", default=0.01)
    parser.add_argument("iterations", default=100)

    args = parser.parse_args()
    train_input_path = args.train_input_path
    w = float(args.initial_weight)
    lr = float(args.learning_rate)
    iters = int(args.iterations)

    data = pd.read_csv(train_input_path)
    X_train = np.array(data['X_train'])
    y_train = np.array(data['y_train'])


    costs = []
    grads = []
    for i in range(iters):
        model = Model(w)
        y_pred = model.forward(X_train)
        grad = gradient(X_train, y_pred, y_train)
        w -= lr*grad

        grads.append(grad)
        costs.append(cost(y_pred, y_train))

    print(costs)