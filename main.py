import numpy as np
from sklearn.datasets import fetch_california_housing

from config import cfg


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

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train():
    pass


def plot():
    pass


def evaluate():
    pass


def main():
    print("Hello from regression-from-scratch!")

    np.random.seed(cfg.seed)

    train_set, val_set, test_set = load_data(cfg.train_split, cfg.val_split)


if __name__ == "__main__":
    main()
