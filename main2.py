import numpy as np
from sklearn.datasets import fetch_california_housing
from tqdm import tqdm

from config import cfg


class LinearRegression:
    def __init__(self, weights=None, bias=1):
        self.weights = None
        self.bias = bias
        self.input = None

        if weights is None:
            self.weights = np.random.uniform(-1, 1)

        print(f"Init with weight: {self.weights}")

    def forward(self, x):
        return x * self.weights + self.bias

    def __call__(self, x):
        return self.forward(x)


class MeanSquaredError:
    def __init__(self):
        self.grad = None

    def forward(y, y_pred):
        return np.mean((y_pred - y) ** 2)

    def __call__(self, y, y_pred):
        return self.forward(y, y_pred)


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

    print(f"Train set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")

    # This line of code pairs each feature set with its corresponding target value for each dataset (train, validation, test)
    # and returns them as lists of tuples. This is done to facilitate easy iteration over the datasets during training and evaluation.
    return (
        list(zip(X_train, y_train)),
        list(zip(X_val, y_val)),
        list(zip(X_test, y_test)),
    )


def normalize_data(data_set):
    """
    Normalize the features in the dataset to have zero mean and unit variance.
    Returns a normalized dataset and the scaling parameters.
    """
    # Extract all features and targets
    all_x = np.array([x for x, _ in data_set])
    all_y = np.array([y for _, y in data_set])

    # Calculate mean and std for each feature
    x_mean = np.mean(all_x, axis=0)
    x_std = np.std(all_x, axis=0)
    y_mean = np.mean(all_y)
    y_std = np.std(all_y)

    # Prevent division by zero
    x_std = np.where(x_std == 0, 1.0, x_std)

    print("Feature means:", x_mean)
    print("Feature std devs:", x_std)
    print(f"Target mean: {y_mean:.4f}, std: {y_std:.4f}")

    # Normalize the data
    normalized_set = []
    for x, y in data_set:
        x_norm = (x - x_mean) / x_std
        y_norm = (y - y_mean) / y_std
        normalized_set.append((x_norm, y_norm))

    scales = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}

    return normalized_set, scales


def train_manual(train_set, val_set, epochs, lr=0.000001):
    n_params = len(train_set[0][0])

    weights = np.random.normal(-0.5, 0.5, n_params)
    # weights = np.zeros(n_params)

    print(f"Initial weights: {weights}")
    bias = 0
    print(f"Initial bias: {bias}")

    # Debug: Let's check a few examples to see the scale of our data
    print("\nData examples:")
    for i in range(min(3, len(train_set))):
        x, y = train_set[i]
        print(
            f"Example {i + 1}: x (min={np.min(x):.4f}, max={np.max(x):.4f}, mean={np.mean(x):.4f}), y={y:.4f}"
        )

    # Check if we need a higher learning rate
    print(f"\nCurrent learning rate: {lr}")
    print("Suggested learning rate range: 0.001 to 0.0001 for this type of data")

    train_loss = []
    # val_loss = []

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        running_loss = 0.0
        nan_encountered = False

        # Debug first 3 iterations of first epoch in detail
        debug_first_epoch = epoch == 0
        debug_count = 0

        for x, y in train_set:
            # Calculate prediction
            y_pred = 0.0
            for i in range(n_params):
                y_pred += weights[i] * x[i]

            y_pred += bias

            # Debug logging for first few iterations
            if debug_first_epoch and debug_count < 3:
                print(f"\nIteration {debug_count + 1} debug:")
                print(f"  x range: {np.min(x):.4f} to {np.max(x):.4f}")
                print(f"  y_true: {y:.4f}, y_pred: {y_pred:.4f}")
                print(f"  prediction error: {y_pred - y:.4f}")

            # Check for extremely large predictions
            if abs(y_pred) > 1e10:
                print(f"WARNING: Very large prediction: {y_pred:.2e}")
                if not nan_encountered:  # Only print details once
                    print(f"  Weights: {weights}")
                    print(f"  Input x: {x}")
                nan_encountered = True

            # Calculate MSE loss with protection against overflow
            error = y_pred - y
            if abs(error) > 1e10:
                print(f"WARNING: Very large error: {error:.2e}")
                # Clip error to prevent overflow
                error = np.clip(error, -1e10, 1e10)

            loss = error**2
            running_loss += loss

            # Debug logging
            if debug_first_epoch and debug_count < 3:
                print(f"  loss: {loss:.6f}")

            # Update weights using gradient descent
            for i in range(n_params):
                # Gradient of MSE with respect to weight
                gradient = 2 * x[i] * error

                # Debug excessive gradients
                if abs(gradient) > 1e10:
                    if debug_first_epoch and debug_count < 3:
                        print(f"  Very large gradient for weight[{i}]: {gradient:.2e}")
                    # Clip gradient to prevent instability
                    gradient = np.clip(gradient, -1e10, 1e10)

                # Apply update with monitoring
                old_weight = weights[i]
                weights[i] = weights[i] - lr * gradient

                if debug_first_epoch and debug_count < 3:
                    print(
                        f"  weight[{i}] update: {old_weight:.6f} -> {weights[i]:.6f}, change: {-lr * gradient:.6f}"
                    )

                # Check for NaN weights
                if np.isnan(weights[i]):
                    print(
                        f"WARNING: NaN detected in weight[{i}] - gradient was {gradient:.2e}"
                    )
                    weights[i] = old_weight  # Revert to old value

            # Update bias
            bias_gradient = 2 * error
            old_bias = bias
            bias = bias - lr * bias_gradient

            if debug_first_epoch and debug_count < 3:
                print(
                    f"  bias update: {old_bias:.6f} -> {bias:.6f}, change: {-lr * bias_gradient:.6f}"
                )
                debug_count += 1

            # Check for NaN in bias
            if np.isnan(bias):
                print(
                    f"WARNING: NaN detected in bias - gradient was {bias_gradient:.2e}"
                )
                bias = old_bias  # Revert to old value

        # Calculate average loss for this epoch
        avg_loss = running_loss / len(train_set)
        train_loss.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
        print(f"  Weights range: {np.min(weights):.6f} to {np.max(weights):.6f}")
        print(f"  Bias: {bias:.6f}")

        # Evaluate on a sample from validation set
        if epoch % 10 == 0 and val_set:
            sample_x, sample_y = val_set[0]
            # Manual calculation to avoid unexpected errors
            sample_pred = 0
            for i in range(n_params):
                sample_pred += weights[i] * sample_x[i]
            sample_pred += bias
            print(
                f"  Sample validation - Prediction: {sample_pred:.4f}, True: {sample_y:.4f}"
            )

    return weights, bias, train_loss


def train(model, loss_fn, train_set, val_set, epochs, lr):
    train_loss = []
    val_loss = []
    print(f"Training for {epochs}")
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        for x, y in train_set:
            y_pred = model(x)

            loss = (y_pred - y) ** 2

            running_loss += loss


def plot():
    pass


def evaluate():
    pass


def main():
    print("Hello from regression-from-scratch!")

    np.random.seed(cfg.seed)

    train_set, val_set, test_set = load_data(cfg.train_split, cfg.val_split)

    # Normalize the data to prevent numerical issues
    print("\nNormalizing data to prevent numerical issues...")
    train_set_norm, scales = normalize_data(train_set)

    # Normalize validation set with same parameters
    val_set_norm = []
    for x, y in val_set:
        x_norm = (x - scales["x_mean"]) / scales["x_std"]
        y_norm = (y - scales["y_mean"]) / scales["y_std"]
        val_set_norm.append((x_norm, y_norm))

    # Use a more appropriate learning rate for normalized data
    learning_rate = 0.01
    print(f"Using learning rate: {learning_rate}")

    model = LinearRegression()
    loss_fn = 0

    weights, bias, train_loss = train_manual(
        train_set_norm, val_set_norm, cfg.epochs, learning_rate
    )

    print(f"Final weights: {weights}")
    print(f"Final bias: {bias}")

    # Convert the weights back to the original scale for interpretation
    print("\nConverting model to original scale...")
    original_weights = weights / scales["x_std"] * scales["y_std"]
    original_bias = (
        bias * scales["y_std"]
        + scales["y_mean"]
        - np.sum(original_weights * scales["x_mean"])
    )

    print(f"Original scale weights: {original_weights}")
    print(f"Original scale bias: {original_bias}")

    # train(model, loss_fn, train_set, val_set, cfg.epochs, cfg.lr)


if __name__ == "__main__":
    main()
