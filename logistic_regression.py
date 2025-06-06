# %% In logistic_regression.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from urllib.request import urlretrieve
import os
import time
from config import cfg


# %% --- Helper functions for data ---


def download_and_load_adult_dataset():
    """Downloads UCI Adult Dataset and returns separate train and test DataFrames."""
    data_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    )
    test_url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    )

    data_path = "data/adult_data.csv"
    test_path = "data/adult_test.csv"

    # Ensure data directory exists
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    if not os.path.exists(data_path):
        print("Downloading adult.data ...")
        urlretrieve(data_url, data_path)
    if not os.path.exists(test_path):
        print("Downloading adult.test ...")
        urlretrieve(test_url, test_path)

    columns = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ]

    try:
        # Training data has no header row and no comment lines at the beginning
        df_train = pd.read_csv(
            data_path, names=columns, na_values="?", skipinitialspace=True
        )
        # Test data has an initial comment line and the target label has a dot at the end
        df_test = pd.read_csv(
            test_path, names=columns, na_values="?", skipinitialspace=True, skiprows=1
        )
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return None, None

    # Correct target label in test set (e.g. ">50K." -> ">50K")
    df_test["income"] = df_test["income"].str.replace(".", "", regex=False)

    return df_train, df_test


def preprocess_adult_data(df_train, df_test):
    """Prepares the Adult DataFrames for logistic regression."""
    print("Starting data processing...")

    # Make copies to avoid modifying original data
    df_train_copy = df_train.copy()
    df_test_copy = df_test.copy()

    # Handle missing values using training data statistics only
    for col in df_train_copy.columns:
        if df_train_copy[col].dtype == "object":
            # Use mode from training data for both train and test
            mode_value = (
                df_train_copy[col].mode()[0]
                if not df_train_copy[col].mode().empty
                else "Unknown"
            )
            df_train_copy[col] = df_train_copy[col].fillna(mode_value)
            df_test_copy[col] = df_test_copy[col].fillna(mode_value)
        else:
            # Use median from training data for both train and test
            median_value = df_train_copy[col].median()
            df_train_copy[col] = df_train_copy[col].fillna(median_value)
            df_test_copy[col] = df_test_copy[col].fillna(median_value)

    # Convert categorical variables to numerical (One-Hot Encoding)
    # Fit encoding on training data only, then apply to both
    categorical_cols = df_train_copy.select_dtypes(include=["object"]).columns.tolist()

    # Handle target variable income separately
    if "income" in categorical_cols:
        categorical_cols.remove("income")

    # Combine train and test temporarily ONLY for consistent one-hot encoding
    # This ensures both datasets have the same columns
    df_combined = pd.concat(
        [df_train_copy, df_test_copy], ignore_index=True, keys=["train", "test"]
    )
    df_encoded = pd.get_dummies(df_combined, columns=categorical_cols, drop_first=True)

    # Split back into train and test
    df_train_encoded = df_encoded.loc["train"].reset_index(drop=True)
    df_test_encoded = df_encoded.loc["test"].reset_index(drop=True)

    # Convert target variable: >50K -> 1, <=50K -> 0
    df_train_encoded["income"] = df_train_encoded["income"].apply(
        lambda x: 1 if x.strip() == ">50K" else 0
    )
    df_test_encoded["income"] = df_test_encoded["income"].apply(
        lambda x: 1 if x.strip() == ">50K" else 0
    )

    # Separate features and targets
    y_train_full = df_train_encoded["income"].values
    y_test = df_test_encoded["income"].values

    X_train_df = df_train_encoded.drop("income", axis=1)
    X_test_df = df_test_encoded.drop("income", axis=1)

    # Ensure all columns are numeric
    for col in X_train_df.columns:
        X_train_df[col] = pd.to_numeric(X_train_df[col], errors="coerce")
        X_test_df[col] = pd.to_numeric(X_test_df[col], errors="coerce")

    # Fill any NaN values that might have been created during conversion
    X_train_df = X_train_df.fillna(0)
    X_test_df = X_test_df.fillna(0)

    X_train_full = X_train_df.values.astype(np.float64)
    X_test = X_test_df.values.astype(np.float64)

    print(
        f"Shape of X_train after One-Hot-Encoding (before normalization): {X_train_full.shape}"
    )
    print(
        f"Shape of X_test after One-Hot-Encoding (before normalization): {X_test.shape}"
    )

    # Split training data into train and validation
    num_train_samples = X_train_full.shape[0]
    np.random.seed(cfg.seed)  # For reproducibility
    train_indices = np.random.permutation(num_train_samples)

    train_end_idx = int(
        cfg.train_split / (cfg.train_split + cfg.val_split) * num_train_samples
    )

    train_idx = train_indices[:train_end_idx]
    val_idx = train_indices[train_end_idx:]

    X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
    X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

    # Normalize features using only training data statistics
    mean_X_train = np.mean(X_train, axis=0)
    std_X_train = np.std(X_train, axis=0)
    std_X_train[std_X_train == 0] = 1  # Avoid division by zero for constant columns

    # Apply normalization to all sets using training statistics
    X_train_normalized = (X_train - mean_X_train) / std_X_train
    X_val_normalized = (X_val - mean_X_train) / std_X_train
    X_test_normalized = (X_test - mean_X_train) / std_X_train

    # Add bias term x0 = 1
    X_train_final = np.concatenate(
        [np.ones((X_train_normalized.shape[0], 1)), X_train_normalized], axis=1
    )
    X_val_final = np.concatenate(
        [np.ones((X_val_normalized.shape[0], 1)), X_val_normalized], axis=1
    )
    X_test_final = np.concatenate(
        [np.ones((X_test_normalized.shape[0], 1)), X_test_normalized], axis=1
    )

    print("Data preprocessing completed.")
    print(f"Shapes: X_train: {X_train_final.shape}, y_train: {y_train.shape}")
    print(f"Shapes: X_val: {X_val_final.shape}, y_val: {y_val.shape}")
    print(f"Shapes: X_test: {X_test_final.shape}, y_test: {y_test.shape}")

    return X_train_final, y_train, X_val_final, y_val, X_test_final, y_test


# --- Core functions for Logistic Regression ---
def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy_loss(y_true, y_pred_proba):
    """Calculates the binary cross-entropy loss."""
    epsilon = 1e-15  # To avoid log(0)
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    m = len(y_true)
    cost = (-1 / m) * np.sum(
        y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba)
    )
    return cost


def calculate_accuracy(y_true, y_pred_proba):
    """Calculates classification accuracy."""
    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    accuracy = np.mean(y_pred_class == y_true)
    return accuracy


# --- Non-vectorized implementation ---
def logistic_regression_gd_non_vectorized(X_train, y_train, X_val, y_val):
    """Logistic regression with non-vectorized gradient descent."""
    m, n = X_train.shape
    theta = np.zeros(n)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    print(
        f"Starting non-vectorized training: LR={cfg.lr_logistic_manual}, Epochs={cfg.epochs_logistic_manual}"
    )

    for epoch in range(cfg.epochs_logistic_manual):
        gradients = np.zeros(n)
        # Predictions for training data in this epoch (for loss and accuracy)
        h_theta_train_epoch = np.zeros(m)

        for i in range(m):  # Loop over each training example
            xi = X_train[i, :]
            yi = y_train[i]

            zi = np.dot(xi, theta)
            h_theta_xi = sigmoid(zi)
            h_theta_train_epoch[i] = h_theta_xi  # Store prediction

            error_i = h_theta_xi - yi

            for j in range(n):  # Loop over each feature
                gradients[j] += error_i * xi[j]

        gradients /= m  # Average of gradients
        theta -= cfg.lr_logistic_manual * gradients

        # Loss and accuracy for training data
        current_train_loss = binary_cross_entropy_loss(y_train, h_theta_train_epoch)
        current_train_acc = calculate_accuracy(y_train, h_theta_train_epoch)
        train_loss_history.append(current_train_loss)
        train_acc_history.append(current_train_acc)

        # Loss and accuracy for validation data
        val_pred_proba = sigmoid(np.dot(X_val, theta))
        current_val_loss = binary_cross_entropy_loss(y_val, val_pred_proba)
        current_val_acc = calculate_accuracy(y_val, val_pred_proba)
        val_loss_history.append(current_val_loss)
        val_acc_history.append(current_val_acc)

        if (epoch + 1) % (
            cfg.epochs_logistic_manual // 10 or 1
        ) == 0:  # Print about 10 times
            print(
                f"  Epoch {epoch + 1}/{cfg.epochs_logistic_manual} - Train Loss: {current_train_loss:.4f}, Val Loss: {current_val_loss:.4f}, Train Acc: {current_train_acc:.4f}, Val Acc: {current_val_acc:.4f}"
            )

    return (
        theta,
        train_loss_history,
        val_loss_history,
        train_acc_history,
        val_acc_history,
    )


# --- Vectorized implementation ---
def logistic_regression_gd_vectorized(X_train, y_train, X_val, y_val):
    """Logistic regression with vectorized gradient descent."""
    m, n = X_train.shape
    theta = np.zeros(n)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    print(
        f"Starting vectorized training: LR={cfg.lr_logistic_full_batch}, Epochs={cfg.epochs_logistic_full_batch}"
    )

    for epoch in range(cfg.epochs_logistic_full_batch):
        z_train = np.dot(X_train, theta)
        h_theta_train = sigmoid(z_train)  # Vectorized prediction

        errors = h_theta_train - y_train
        gradients = (1 / m) * np.dot(X_train.T, errors)  # Vectorized gradient

        theta -= cfg.lr_logistic_full_batch * gradients

        # Loss and accuracy for training data
        current_train_loss = binary_cross_entropy_loss(y_train, h_theta_train)
        current_train_acc = calculate_accuracy(y_train, h_theta_train)
        train_loss_history.append(current_train_loss)
        train_acc_history.append(current_train_acc)

        # Loss and accuracy for validation data
        val_pred_proba = sigmoid(np.dot(X_val, theta))
        current_val_loss = binary_cross_entropy_loss(y_val, val_pred_proba)
        current_val_acc = calculate_accuracy(y_val, val_pred_proba)
        val_loss_history.append(current_val_loss)
        val_acc_history.append(current_val_acc)

        if (epoch + 1) % (
            cfg.epochs_logistic_full_batch // 10 or 1
        ) == 0:  # Print about 10 times
            print(
                f"  Epoch {epoch + 1}/{cfg.epochs_logistic_full_batch} - Train Loss: {current_train_loss:.4f}, Val Loss: {current_val_loss:.4f}, Train Acc: {current_train_acc:.4f}, Val Acc: {current_val_acc:.4f}"
            )

    return (
        theta,
        train_loss_history,
        val_loss_history,
        train_acc_history,
        val_acc_history,
    )


# --- Plotting and evaluation functions ---
def plot_learning_curves(train_hist, val_hist, title, ylabel, filename_suffix):
    plt.figure(figsize=(10, 6))
    plt.plot(train_hist, label=f"Training {ylabel}")
    plt.plot(val_hist, label=f"Validation {ylabel}")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    filepath = os.path.join(cfg.results_dir_logistic, f"{filename_suffix}.png")
    plt.savefig(filepath)
    print(f"Plot saved: {filepath}")
    plt.close()


def evaluate_model_on_test_set(theta, X_test, y_test):
    """Evaluates the model on the test set and outputs metrics."""
    y_pred_proba_test = sigmoid(np.dot(X_test, theta))
    y_pred_class_test = (y_pred_proba_test >= 0.5).astype(int)

    # Calculate metrics
    tp = np.sum((y_test == 1) & (y_pred_class_test == 1))
    tn = np.sum((y_test == 0) & (y_pred_class_test == 0))
    fp = np.sum((y_test == 0) & (y_pred_class_test == 1))
    fn = np.sum((y_test == 1) & (y_pred_class_test == 0))

    accuracy = (tp + tn) / len(y_test) if len(y_test) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print(f"  Test Accuracy: {accuracy:.4f}")
    print(f"  Test Precision: {precision:.4f}")
    print(f"  Test Recall: {recall:.4f}")
    print(f"  Test F1-Score: {f1_score:.4f}")

    # Plot Confusion Matrix
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred <=50K", "Pred >50K"])
    plt.yticks(tick_marks, ["Actual <=50K", "Actual >50K"])

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    filepath = os.path.join(
        cfg.results_dir_logistic, "confusion_matrix_test.png"
    )  # Single file for both versions if theta is similar
    # Could also save separate CMs for non-vec and vec if desired
    plt.savefig(filepath)
    print(f"Confusion Matrix saved: {filepath}")
    plt.close()

    return accuracy, precision, recall, f1_score


# %% --- Main execution block ---

print("Starting Logistic Regression Assignment...")

# Ensure results directory exists
os.makedirs(cfg.results_dir_logistic, exist_ok=True)

# Load and prepare data
df_train, df_test = download_and_load_adult_dataset()
# %%

X_train, y_train, X_val, y_val, X_test, y_test = preprocess_adult_data(
    df_train, df_test
)


# %% --- Non-vectorized implementation ---
print("\n--- Training Non-Vectorized Logistic Regression ---")
start_time_non_vec = time.time()
theta_nv, tl_nv, vl_nv, ta_nv, va_nv = logistic_regression_gd_non_vectorized(
    X_train, y_train, X_val, y_val
)
end_time_non_vec = time.time()
print(
    f"Training time (Non-Vectorized): {end_time_non_vec - start_time_non_vec:.2f} seconds"
)

plot_learning_curves(
    tl_nv,
    vl_nv,
    "Non-Vectorized: Loss vs. Epochs",
    "Loss (Binary Cross-Entropy)",
    "loss_non_vectorized_logistic",
)
plot_learning_curves(
    ta_nv,
    va_nv,
    "Non-Vectorized: Accuracy vs. Epochs",
    "Accuracy",
    "accuracy_non_vectorized_logistic",
)

print("\nEvaluation of non-vectorized model on test set:")
evaluate_model_on_test_set(theta_nv, X_test, y_test)

# --- Vectorized implementation ---
print("\n--- Training Vectorized Logistic Regression ---")
start_time_vec = time.time()
theta_v, tl_v, vl_v, ta_v, va_v = logistic_regression_gd_vectorized(
    X_train, y_train, X_val, y_val
)
end_time_vec = time.time()
print(f"Training time (Vectorized): {end_time_vec - start_time_vec:.2f} seconds")

plot_learning_curves(
    tl_v,
    vl_v,
    "Vectorized: Loss vs. Epochs",
    "Loss (Binary Cross-Entropy)",
    "loss_vectorized_logistic",
)
plot_learning_curves(
    ta_v,
    va_v,
    "Vectorized: Accuracy vs. Epochs",
    "Accuracy",
    "accuracy_vectorized_logistic",
)

print("\nEvaluation of vectorized model on test set:")
# We can use the same function since the theta of the vectorized version is probably better.
# If you want to evaluate both thetas separately, you can name the CM differently.
evaluate_model_on_test_set(theta_v, X_test, y_test)

print("\nComparison of training times:")
print(f"  Non-Vectorized: {end_time_non_vec - start_time_non_vec:.2f}s")
print(f"  Vectorized:     {end_time_vec - start_time_vec:.2f}s")

print(
    f"\nAll results and plots of the logistic regression have been saved in '{cfg.results_dir_logistic}'."
)
print("Logistic Regression Assignment part completed.")
