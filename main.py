import numpy as np
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import cfg


class LinearRegression:
    def __init__(self, n_params, weights=None, bias=1, lr=0.001, full_batch=False):
        self.input = None
        self.bias = bias
        self.weights = weights
        self.lr = lr
        self.full_batch = full_batch

        if weights is None:
            print("Initial weights are None, generating random weights")
            self.weights = np.random.uniform(-1, 1, n_params)

    def forward(self, x, no_grad=False):
        if not no_grad:
            self.input = x
        if self.full_batch:
            return x @ self.weights + self.bias
        else:
            # check if not also @
            return np.dot(x, self.weights) + self.bias

    def step(self, grad):
        if self.full_batch:
            grad_w = self.input.T @ grad
            # / len(self.input)
            grad_b = np.mean(grad)

            # Clip gradients to prevent overflow
            grad_w = np.clip(grad_w, -10, 10)
            grad_b = np.clip(grad_b, -10, 10)

            self.weights = self.weights - self.lr * grad_w
            self.bias = self.bias - self.lr * grad_b
        else:
            self.weights = self.weights - self.lr * grad * self.input
            self.bias = self.bias - self.lr * grad

    def __call__(self, x, no_grad=False):
        return self.forward(x, no_grad)


class MSE:
    def __init__(self, full_batch=False):
        self.y_pred = None
        self.y = None
        self.full_batch = full_batch

    def forward(self, y_pred, y, no_grad=False):
        if not no_grad:
            self.y_pred = y_pred
            self.y = y
        if self.full_batch:
            return np.mean((y_pred - y) ** 2)
        else:
            return (y_pred - y) ** 2

    def backward(self):
        if self.full_batch:
            grad = 2 * (self.y_pred - self.y)
            # / len(self.y)
        else:
            grad = 2 * (self.y_pred - self.y)
        return grad

    def __call__(self, y_pred, y, no_grad=False):
        return self.forward(y_pred, y, no_grad)


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

    # divide the results by

    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    return (
        list(zip(X_train_normed, y_train_normed)),
        list(zip(X_val_normed, y_val_normed)),
        list(zip(X_test_normed, y_test_normed)),
    )


def train_manual(train_set, val_set, epochs, lr, weights=None, bias=1, verbose=False):
    n_params = len(train_set[0][0])

    # init weights and bias if not provided
    if weights is None:
        weights = np.random.uniform(-1, 1, n_params)
    if bias is None:
        bias = 1

    train_loss_list = []
    val_loss_list = []

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

            # calculate grad of loss
            loss_grad = 2 * (y_pred - y)

            # update weights
            for i in range(n_params):
                weight_grad = x[i] * loss_grad
                weights[i] = weights[i] - lr * weight_grad

            # update bias
            bias_grad = loss_grad
            bias = bias - lr * bias_grad

        normalised_loss = running_loss / len(train_set)
        train_loss_list.append(normalised_loss)

        # evaluate on validation set
        val_loss = evaluate_model_manual(weights, bias, val_set)
        val_loss_list.append(val_loss)

        if verbose:
            # make sample prediction
            x, y = train_set[0]
            y_pred = 0.0
            for i in range(n_params):
                y_pred += weights[i] * x[i]
            y_pred = y_pred + bias
            print(f"Epoch {epoch}: Prediction: {y_pred:.4f}, True Value: {y:.4f}")

    return train_loss_list, val_loss_list


def train_vectorised(model, loss_fn, train_set, val_set, epochs, verbose=False):
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

        if verbose:
            # make sample prediction
            x, y = train_set[0]
            y_pred = model(x)
            print(f"Epoch {epoch}: Prediction: {y_pred:.4f}, True Value: {y:.4f}")

    return train_loss


def train_vectorised_full_batch(
    model, loss_fn, train_set, val_set, epochs, verbose=False
):
    train_loss_list = []
    val_loss_list = []

    # convert train_set to numpy arrays to process in parallel
    x_train = np.array([item[0] for item in train_set])
    y_train = np.array([item[1] for item in train_set])

    x_val = np.array([item[0] for item in val_set])
    y_val = np.array([item[1] for item in val_set])

    for epoch in tqdm(range(epochs)):
        # make prediction
        y_pred = model(x_train)
        # calculate loss
        loss = loss_fn(y_pred, y_train)
        # get grad of loss function
        grad = loss_fn.backward()
        # update model
        model.step(grad)
        # evaluate model on validation set
        val_loss = evaluate_model_full(model, loss_fn, x_val, y_val)

        # log loss
        normalised_train_loss = loss
        train_loss_list.append(normalised_train_loss)
        normalised_val_loss = val_loss
        val_loss_list.append(normalised_val_loss)

        if verbose:
            # make sample prediction
            x, y = x_train[0], y_train[0]
            y_pred = model(x)
            print(f"Epoch {epoch}: Prediction: {y_pred:.4f}, True Value: {y:.4f}")

    return train_loss_list, val_loss_list


def evaluate_model_full(model, loss_fn, x_test, y_test):
    # make prediction
    y_pred = model(x_test, no_grad=True)
    # calculate loss
    loss = loss_fn(y_pred, y_test, no_grad=True)
    return loss


def evaluate_model_manual(weights, bias, test_set):
    running_loss = 0.0
    # Iterate over each data point in the test set
    for x, y in test_set:
        y_pred = 0.0
        # Calculate prediction using weights and bias
        for i in range(len(weights)):
            y_pred += weights[i] * x[i]
        y_pred = y_pred + bias
        # Calculate loss for this data point
        loss = (y_pred - y) ** 2
        running_loss += loss

    # Calculate normalised loss by dividing by the number of data points
    normalised_loss = running_loss / len(test_set)
    return normalised_loss


def plot_train_vs_validation_loss(train_loss, val_loss, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label="Training Loss", color="blue")
    plt.plot(val_loss, label="Validation Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"results/{filename}.png")
    plt.close()  # Close the figure to free memory


def plot_predictions_vs_actual(test_set, weights_manual, bias_manual, model):
    x_test = np.array([item[0] for item in test_set])
    y_test = np.array([item[1] for item in test_set])

    # Manual model predictions
    manual_predictions = []
    for x in x_test:
        y_pred = 0.0
        for i in range(len(weights_manual)):
            y_pred += weights_manual[i] * x[i]
        y_pred = y_pred + bias_manual
        manual_predictions.append(y_pred)
    manual_predictions = np.array(manual_predictions)

    # Vectorised model predictions
    vectorised_predictions = model(x_test, no_grad=True)

    # Plot predictions vs actual values
    plt.figure(figsize=(10, 6))

    # Sort values for better visualization
    sort_idx = np.argsort(y_test)
    y_test_sorted = y_test[sort_idx]
    manual_pred_sorted = manual_predictions[sort_idx]
    vectorised_pred_sorted = vectorised_predictions[sort_idx]

    # Plot predictions first (so they appear behind actual values)
    plt.scatter(
        range(len(manual_pred_sorted)),
        manual_pred_sorted,
        label="Manual Predictions",
        alpha=0.5,
        color="blue",
        zorder=1,
    )
    plt.scatter(
        range(len(vectorised_pred_sorted)),
        vectorised_pred_sorted,
        label="Vectorised Predictions",
        alpha=0.5,
        color="red",
        zorder=1,
    )

    # Plot actual values last (so they appear on top)
    plt.scatter(
        range(len(y_test_sorted)),
        y_test_sorted,
        label="Actual Values",
        alpha=0.5,
        color="black",
        zorder=2,
    )

    plt.xlabel("Sample Index (sorted by actual values)")
    plt.ylabel("Value")
    plt.title("Manual vs Vectorised Predictions on Test Set")
    plt.legend()
    plt.savefig("results/manual_vs_vectorised_predictions_plot.png")


def main():
    VERBOSE = False

    # Set random seed for reproducibility
    np.random.seed(cfg.seed)

    train_set, val_set, test_set = load_data(cfg.train_split, cfg.val_split)

    n_params = len(train_set[0][0])

    # Initialize the same random weights, to ensure fair comparison
    weights_manual = np.random.uniform(-1, 1, n_params)
    bias_manual = 1

    # Copy manual weights for vectorized training to ensure fair comparison
    weights_vectorised = weights_manual.copy()
    bias_vectorised = 1

    # Initialize model
    model = LinearRegression(
        n_params=n_params,
        lr=cfg.lr_full_batch,
        weights=weights_vectorised,
        bias=bias_vectorised,
        full_batch=True,
    )
    loss_fn = MSE(full_batch=True)

    print("Training manual model...")
    train_loss_manual, val_loss_manual = train_manual(
        train_set,
        val_set,
        cfg.epochs_manual,
        cfg.lr_manual,
        weights_manual,
        bias_manual,
        verbose=VERBOSE,
    )

    print("Training vectorised model...")
    train_loss_vectorised, val_loss_vectorised = train_vectorised_full_batch(
        model, loss_fn, train_set, val_set, cfg.epochs_full_batch, verbose=VERBOSE
    )

    print("Evaluating manual model...")
    test_loss_manual = evaluate_model_manual(weights_manual, bias_manual, test_set)

    print("Evaluating vectorised model...")
    x_test = np.array([item[0] for item in test_set])
    y_test = np.array([item[1] for item in test_set])
    test_loss_vectorised = evaluate_model_full(model, loss_fn, x_test, y_test)

    print(f"Manual training loss: {train_loss_manual[-1]:.4f}")
    print(f"Manual validation loss: {val_loss_manual[-1]:.4f}")
    print(f"Manual test loss: {test_loss_manual:.4f}")

    print(f"Vectorised training loss: {train_loss_vectorised[-1]:.4f}")
    print(f"Vectorised validation loss: {val_loss_vectorised[-1]:.4f}")
    print(f"Vectorised test loss: {test_loss_vectorised:.4f}")

    # print number of update steps per model
    print("Manual update steps:", len(train_loss_manual) * len(train_set))
    print("Vectorised update steps:", cfg.epochs_full_batch)

    plot_predictions_vs_actual(test_set, weights_manual, bias_manual, model)

    # Plot training vs validation losses for each method separately
    plot_train_vs_validation_loss(
        train_loss_manual,
        val_loss_manual,
        "Manual Implementation: Training vs Validation Loss",
        "manual_train_vs_val_loss",
    )
    plot_train_vs_validation_loss(
        train_loss_vectorised,
        val_loss_vectorised,
        "Vectorised Implementation: Training vs Validation Loss",
        "vectorised_train_vs_val_loss",
    )


if __name__ == "__main__":
    main()
