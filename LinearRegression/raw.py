import Model

n_samples = 100
n_epochs = 30

# sample data
X_train = [i for i in range(n_samples)]
y_train = [3*X_train[i] for i in range(n_samples)]

# cost
def cost(y_pred: float, y_true: float) -> float:
    cost = 0
    for i in range(n_samples):
        cost += ((y_pred[i] - y_true[i])**2)/n_samples
    return cost

# gradient
def gradient(w: float, cost: float) -> float:
    grad = 0
    model = Model(w)
    for i in range(n_samples):
        grad += 2 * (model.forward(X_train[i]) - y_train[i]) * X_train[i]
    grad = grad / n_samples
    return grad
    
for i in range(n_samples):
    