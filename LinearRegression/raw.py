n_samples = 100

# sample data
X_train = [i for i in range(n_samples)]
y_train = [3*X_train[i] for i in range(n_samples)]

# cost
def cost(y_pred, y_true):
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
    w = 4
    iters = 100
    
    for i in range(iters):
        w -= gradient(w, cost())