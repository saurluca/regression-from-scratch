import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from config import cfg


class LinearRegression:
    def __init__(self, n_params, weights=None, bias=1, lr=0.001):
        self.input = None
        self.bias = bias
        self.weights = weights
        self.lr = lr

        print(f"MODEL Initial weights: {weights}")
        print(f"Initial bias: {bias}")
        print(f"Learning rate: {lr}")

        if weights is None:
            print("Initial weights are None, generating random weights")
            self.weights = np.random.uniform(-1, 1, n_params)

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def step(self, grad):
        self.weights = self.weights - self.lr * grad * self.input
        self.bias = self.bias - self.lr * grad

    def __call__(self, x):
        return self.forward(x)


class MSE:
    def __init__(self):
        self.y_pred = None
        self.y = None

    def forward(self, y_pred, y):
        self.y_pred = y_pred
        self.y = y
        return np.mean((y_pred - y) ** 2)

    def backward(self):
        grad = (
            2 * (self.y_pred - self.y) / len(self.y)
            if hasattr(self.y, "__len__")
            else 2 * (self.y_pred - self.y)
        )
        return grad

    def __call__(self, y_pred, y):
        return self.forward(y_pred, y)


def load_data(train_split, val_split):
    # Get features (X) and targets (y) from the California housing dataset
    X, y = fetch_california_housing(return_X_y=True)

    # Use np.random.permutation to shuffle indices
    indices = np.random.permutation(len(X))

    # Use the same shuffled indices for both features and targets to maintain alignment
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # check if data is actually shuffled
    assert not np.array_equal(X_shuffled[0], X[0])

    # Calculate split sizes
    num_samples = len(X)
    num_train = int(num_samples * train_split)
    num_val = int(num_samples * val_split)

    # Split the data into train, validation, and test sets
    X_train = X_shuffled[:num_train]
    y_train = y_shuffled[:num_train]

    X_val = X_shuffled[num_train : num_train + num_val]
    y_val = y_shuffled[num_train : num_train + num_val]

    X_test = X_shuffled[num_train + num_val :]
    y_test = y_shuffled[num_train + num_val :]

    # Compute normalization parameters from training data only
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    y_train_mean = np.mean(y_train)
    y_train_std = np.std(y_train)

    # Normalize all sets using training set statistics
    X_train_normed = (X_train - X_train_mean) / X_train_std
    X_val_normed = (X_val - X_train_mean) / X_train_std
    X_test_normed = (X_test - X_train_mean) / X_train_std

    y_train_normed = (y_train - y_train_mean) / y_train_std
    y_val_normed = (y_val - y_train_mean) / y_train_std
    y_test_normed = (y_test - y_train_mean) / y_train_std

    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    return (
        list(zip(X_train_normed, y_train_normed)),
        list(zip(X_val_normed, y_val_normed)),
        list(zip(X_test_normed, y_test_normed)),
    )


def train_manual(train_set, val_set, epochs, lr, weights=None, bias=1):
    n_params = len(train_set[0][0])

    if weights is None:
        weights = np.random.uniform(-1, 1, n_params)
        
    if bias is None:
        bias = 1

    train_loss = []

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for x, y in train_set:
            # make linear prediction
            y_pred = 0.0
            for i in range(n_params):
                y_pred += weights[i] * x[i]

            y_pred = y_pred + bias
            
            # calculate MSE
            loss = (y_pred - y) ** 2
            running_loss += loss
            
            loss_grad = 2 * (y_pred - y)

            # update weights
            for i in range(n_params):
                weight_grad = x[i] * loss_grad
                weights[i] = weights[i] - lr * weight_grad

            # update bias
            bias_grad = loss_grad
            bias = bias - lr * bias_grad

        normalised_loss = running_loss / len(train_set)
        train_loss.append(normalised_loss)

        # make sample prediction
        x, y = train_set[0]
        y_pred = 0.0
        for i in range(n_params):
            y_pred += weights[i] * x[i]
        y_pred = y_pred + bias
        # print(f"Epoch {epoch}: Prediction: {y_pred:.4f}, True Value: {y:.4f}")

    return train_loss


def train_vectorised(model, loss_fn, train_set, val_set, epochs):
    train_loss = []
    # val_loss = []

    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for x, y in train_set:
            # make prediction
            y_pred = model(x)

            # calculate loss
            loss = loss_fn(y_pred, y)
            running_loss += loss

            # get grad of loss function
            grad = loss_fn.backward()

            # update model
            model.step(grad)

        normalised_loss = running_loss / len(train_set)
        train_loss.append(normalised_loss)

        # make sample prediction
        x, y = train_set[0]
        y_pred = model(x)
        # print(f"Epoch {epoch}: Prediction: {y_pred:.4f}, True Value: {y:.4f}")

    return train_loss


def plot_results(train_loss):
    plt.plot(train_loss, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.savefig("results/training_loss_plot.png")


def main():
    # Set random seed for reproducibility
    np.random.seed(cfg.seed)

    train_set, val_set, test_set = load_data(cfg.train_split, cfg.val_split)

    n_params = len(train_set[0][0])
    
    # Initialize random weights for manual training
    weights_manual = np.random.uniform(-1, 1, n_params)
    bias_manual = 1
    
    # Copy manual weights for vectorized training to ensure fair comparison
    weights_vectorised = weights_manual.copy()
    bias_vectorised = 1

    # Initialize model
    model = LinearRegression(n_params=n_params, lr=cfg.lr, weights=weights_vectorised, bias=bias_vectorised)    
    loss_fn = MSE()

    print("Training manual model...")
    start_time = time.time()
    train_loss_manual = train_manual(train_set, val_set, cfg.epochs, cfg.lr, weights_manual, bias_manual)
    manual_time = time.time() - start_time
    print(f"Manual training took {manual_time:.2f} seconds")

    print("Training vectorised model...")
    start_time = time.time()
    train_loss_vectorised = train_vectorised(model, loss_fn, train_set, val_set, cfg.epochs)
    vectorised_time = time.time() - start_time
    print(f"Vectorised training took {vectorised_time:.2f} seconds")
    
    
    print("Manual training loss:", train_loss_manual[-1])
    print("Vectorised training loss:", train_loss_vectorised[-1])
    
    plt.figure(figsize=(10,6))
    plt.plot(train_loss_manual, label="Manual Training Loss")
    plt.plot(train_loss_vectorised, label="Vectorised Training Loss") 
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.savefig("results/training_loss_plot.png")


if __name__ == "__main__":
    main()
