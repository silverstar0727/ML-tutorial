import argparse

# cost
def cost(n_samples, y_pred, y_true):
    cost = 0
    for i in range(n_samples):
        cost += ((y_pred[i] - y_true[i])**2)/n_samples
    return cost

# define the model
class Model():
    def __init__(self, w):
       self.w = w
    
    def forward(self, input):
        return self.w*input 

# gradient
def gradient(w, cost , X, y, n_samples):
    grad = 0
    model = Model(w)
    for i in range(n_samples):
        grad += 2 * (model.forward(X[i]) - y[i]) * X[i]
    grad = grad / n_samples
    cost()
    return grad, cost
    
if __init__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_samples", default=100)
    parser.add_argument("train_input_path", default=None)
    parser.add_argument("initial_weight", default=4)
    parser.add_argument("learning_rate", default=0.01)
    parser.add_argument("iterations", default=100)

    args = parser.parse_args()
    n_samples = args.n_samples
    train_input_path = args.train_input_path
    w = args.initial_weight
    lr = args.learning_rate
    iters = args.iterations

    if train_input_path == None:
        # sample data
        X_train = [i for i in range(n_samples)]
        y_train = [3*X_train[i] for i in range(n_samples)]

    for i in range(iters):
        w -= gradient(w, cost())