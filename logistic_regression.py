# In logistic_regression.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from urllib.request import urlretrieve
import os
import time
from config import cfg


# --- Hilfsfunktionen für Daten ---

def download_and_load_adult_dataset():
    """Downloads UCI Adult Dataset und als Pandas DataFrame."""
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

    data_path = "adult.data"
    test_path = "adult.test"

    if not os.path.exists(data_path):
        print("Download adult.data ...")
        urlretrieve(data_url, data_path)
    if not os.path.exists(test_path):
        print("Download adult.test ...")
        urlretrieve(test_url, test_path)

    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    try:
        # Trainingsdaten haben keine Header-Zeile und keine Kommentarzeilen am Anfang
        df_train = pd.read_csv(data_path, names=columns, na_values="?", skipinitialspace=True)
        # Testdaten haben eine initiale Kommentarzeile und das Ziel-Label hat einen Punkt am Ende
        df_test = pd.read_csv(test_path, names=columns, na_values="?", skipinitialspace=True, skiprows=1)
    except Exception as e:
        print(f"Fehler beim Lesen der CSV-Dateien: {e}")
        return None

    # Korrigiere Ziel-Label im Testset (z.B. ">50K." -> ">50K")
    df_test['income'] = df_test['income'].str.replace('.', '', regex=False)

    # Kombiniere für konsistente Vorverarbeitung (insbesondere One-Hot-Encoding)
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    return df_combined


def preprocess_adult_data(df):
    """Bereitet den Adult DataFrame für die logistische Regression vor."""
    print("Start Dataprocessing...")
    # 1. Fehlende Werte behandeln
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)  # Modus für kategoriale
        else:
            df[col].fillna(df[col].median(), inplace=True)  # Median für numerische

    # 2. Kategoriale Variablen in numerische umwandeln (One-Hot Encoding)
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Zielvariable 'income' ist auch 'object', behandle sie separat
    if 'income' in categorical_cols:
        categorical_cols.remove('income')

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # 3. Zielvariable umwandeln: >50K -> 1, <=50K -> 0
    df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

    y = df['income'].values
    X = df.drop('income', axis=1).values
    print(f"Form von X nach One-Hot-Encoding (vor Normalisierung): {X.shape}")

    # 4. Kontinuierliche Features normalisieren (alle Spalten in X sind jetzt numerisch)
    #    Wichtig: Normalisierungsparameter (mean, std) nur auf Trainingsdaten berechnen
    #    und dann auf Validierungs-/Testdaten anwenden.
    #    Für die Einfachheit dieses Assignments (und analog zum linearen Regressionsbeispiel)
    #    können wir hier auf dem gesamten X vor dem Split normalisieren,
    #    aber in der Praxis ist das anders.
    mean_X = np.mean(X, axis=0)
    std_X = np.std(X, axis=0)
    std_X[std_X == 0] = 1  # Vermeide Division durch Null für konstante Spalten
    X_normalized = (X - mean_X) / std_X

    # 5. Bias-Term x0 = 1 hinzufügen
    X_final = np.concatenate([np.ones((X_normalized.shape[0], 1)), X_normalized], axis=1)
    print(f"Form von X nach Hinzufügen des Bias-Terms: {X_final.shape}")

    # 6. Daten mischen und aufteilen
    num_samples = X_final.shape[0]
    np.random.seed(cfg.seed)  # Für Reproduzierbarkeit
    indices = np.random.permutation(num_samples)

    X_shuffled = X_final[indices]
    y_shuffled = y[indices]

    train_end_idx = int(cfg.train_split * num_samples)
    val_end_idx = train_end_idx + int(cfg.val_split * num_samples)

    X_train, y_train = X_shuffled[:train_end_idx], y_shuffled[:train_end_idx]
    X_val, y_val = X_shuffled[train_end_idx:val_end_idx], y_shuffled[train_end_idx:val_end_idx]
    X_test, y_test = X_shuffled[val_end_idx:], y_shuffled[val_end_idx:]

    print("Datenvorverarbeitung abgeschlossen.")
    print(f"Shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Shapes: X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"Shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# --- Kernfunktionen für Logistische Regression ---

def sigmoid(z):
    """Sigmoid-Aktivierungsfunktion."""
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy_loss(y_true, y_pred_proba):
    """Berechnet den binären Kreuzentropie-Verlust."""
    epsilon = 1e-15  # Um log(0) zu vermeiden
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
    m = len(y_true)
    cost = (-1 / m) * np.sum(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
    return cost


def calculate_accuracy(y_true, y_pred_proba):
    """Berechnet die Klassifikationsgenauigkeit."""
    y_pred_class = (y_pred_proba >= 0.5).astype(int)
    accuracy = np.mean(y_pred_class == y_true)
    return accuracy


# --- Nicht-vektorisierte Implementierung ---
def logistic_regression_gd_non_vectorized(X_train, y_train, X_val, y_val):
    """Logistische Regression mit nicht-vektorisiertem Gradientenabstieg."""
    m, n = X_train.shape
    theta = np.zeros(n)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    print(f"Starte nicht-vektorisiertes Training: LR={cfg.lr_logistic_manual}, Epochen={cfg.epochs_logistic_manual}")

    for epoch in range(cfg.epochs_logistic_manual):
        gradients = np.zeros(n)
        # Vorhersagen für Trainingsdaten in dieser Epoche (für Verlust und Genauigkeit)
        h_theta_train_epoch = np.zeros(m)

        for i in range(m):  # Schleife über jedes Trainingsbeispiel
            xi = X_train[i, :]
            yi = y_train[i]

            zi = np.dot(xi, theta)
            h_theta_xi = sigmoid(zi)
            h_theta_train_epoch[i] = h_theta_xi  # Speichere Vorhersage

            error_i = h_theta_xi - yi

            for j in range(n):  # Schleife über jedes Feature
                gradients[j] += error_i * xi[j]

        gradients /= m  # Mittelwert der Gradienten
        theta -= cfg.lr_logistic_manual * gradients

        # Verlust und Genauigkeit für Trainingsdaten
        current_train_loss = binary_cross_entropy_loss(y_train, h_theta_train_epoch)
        current_train_acc = calculate_accuracy(y_train, h_theta_train_epoch)
        train_loss_history.append(current_train_loss)
        train_acc_history.append(current_train_acc)

        # Verlust und Genauigkeit für Validierungsdaten
        val_pred_proba = sigmoid(np.dot(X_val, theta))
        current_val_loss = binary_cross_entropy_loss(y_val, val_pred_proba)
        current_val_acc = calculate_accuracy(y_val, val_pred_proba)
        val_loss_history.append(current_val_loss)
        val_acc_history.append(current_val_acc)

        if (epoch + 1) % (cfg.epochs_logistic_manual // 10 or 1) == 0:  # Ca. 10 Mal ausgeben
            print(
                f"  Epoch {epoch + 1}/{cfg.epochs_logistic_manual} - Train Loss: {current_train_loss:.4f}, Val Loss: {current_val_loss:.4f}, Train Acc: {current_train_acc:.4f}, Val Acc: {current_val_acc:.4f}")

    return theta, train_loss_history, val_loss_history, train_acc_history, val_acc_history


# --- Vektorisierte Implementierung ---
def logistic_regression_gd_vectorized(X_train, y_train, X_val, y_val):
    """Logistische Regression mit vektorisiertem Gradientenabstieg."""
    m, n = X_train.shape
    theta = np.zeros(n)

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    print(f"Starte vektorisiertes Training: LR={cfg.lr_logistic_full_batch}, Epochen={cfg.epochs_logistic_full_batch}")

    for epoch in range(cfg.epochs_logistic_full_batch):
        z_train = np.dot(X_train, theta)
        h_theta_train = sigmoid(z_train)  # Vektorisierte Vorhersage

        errors = h_theta_train - y_train
        gradients = (1 / m) * np.dot(X_train.T, errors)  # Vektorisierter Gradient

        theta -= cfg.lr_logistic_full_batch * gradients

        # Verlust und Genauigkeit für Trainingsdaten
        current_train_loss = binary_cross_entropy_loss(y_train, h_theta_train)
        current_train_acc = calculate_accuracy(y_train, h_theta_train)
        train_loss_history.append(current_train_loss)
        train_acc_history.append(current_train_acc)

        # Verlust und Genauigkeit für Validierungsdaten
        val_pred_proba = sigmoid(np.dot(X_val, theta))
        current_val_loss = binary_cross_entropy_loss(y_val, val_pred_proba)
        current_val_acc = calculate_accuracy(y_val, val_pred_proba)
        val_loss_history.append(current_val_loss)
        val_acc_history.append(current_val_acc)

        if (epoch + 1) % (cfg.epochs_logistic_full_batch // 10 or 1) == 0:  # Ca. 10 Mal ausgeben
            print(
                f"  Epoch {epoch + 1}/{cfg.epochs_logistic_full_batch} - Train Loss: {current_train_loss:.4f}, Val Loss: {current_val_loss:.4f}, Train Acc: {current_train_acc:.4f}, Val Acc: {current_val_acc:.4f}")

    return theta, train_loss_history, val_loss_history, train_acc_history, val_acc_history


# --- Plotting und Evaluierungsfunktionen ---
def plot_learning_curves(train_hist, val_hist, title, ylabel, filename_suffix):
    plt.figure(figsize=(10, 6))
    plt.plot(train_hist, label=f'Training {ylabel}')
    plt.plot(val_hist, label=f'Validation {ylabel}')
    plt.title(title)
    plt.xlabel('Epochen')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    filepath = os.path.join(cfg.results_dir_logistic, f"{filename_suffix}.png")
    plt.savefig(filepath)
    print(f"Plot gespeichert: {filepath}")
    plt.close()


def evaluate_model_on_test_set(theta, X_test, y_test):
    """Evaluiert das Modell auf dem Testset und gibt Metriken aus."""
    y_pred_proba_test = sigmoid(np.dot(X_test, theta))
    y_pred_class_test = (y_pred_proba_test >= 0.5).astype(int)

    # Metriken berechnen
    tp = np.sum((y_test == 1) & (y_pred_class_test == 1))
    tn = np.sum((y_test == 0) & (y_pred_class_test == 0))
    fp = np.sum((y_test == 0) & (y_pred_class_test == 1))
    fn = np.sum((y_test == 1) & (y_pred_class_test == 0))

    accuracy = (tp + tn) / len(y_test) if len(y_test) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  Test Genauigkeit: {accuracy:.4f}")
    print(f"  Test Präzision: {precision:.4f}")
    print(f"  Test Recall: {recall:.4f}")
    print(f"  Test F1-Score: {f1_score:.4f}")

    # Confusion Matrix Plotten
    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Test Set)')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred <=50K", "Pred >50K"])
    plt.yticks(tick_marks, ["Actual <=50K", "Actual >50K"])

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Wahre Klasse')
    plt.xlabel('Vorhergesagte Klasse')
    filepath = os.path.join(cfg.results_dir_logistic,
                            "confusion_matrix_test.png")  # Einmalig für beide Versionen, wenn Theta ähnlich
    # Man könnte auch separate CMs für non-vec und vec speichern, wenn man will
    plt.savefig(filepath)
    print(f"Confusion Matrix gespeichert: {filepath}")
    plt.close()

    return accuracy, precision, recall, f1_score


# --- Hauptausführungsblock ---
if __name__ == "__main__":
    print("Starte Logistische Regression Assignment...")

    # Sicherstellen, dass der Ergebnisordner existiert
    if not os.path.exists(cfg.results_dir_logistic):
        os.makedirs(cfg.results_dir_logistic)
        print(f"Ergebnisordner erstellt: {cfg.results_dir_logistic}")

    # 1. Daten laden und vorbereiten
    df_adult = download_and_load_adult_dataset()
    if df_adult is None:
        print("Fehler beim Laden des Datensatzes. Programm wird beendet.")
        exit()

    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_adult_data(df_adult)

    # --- Nicht-vektorisierte Implementierung ---
    print("\n--- Training Nicht-Vektorisierte Logistische Regression ---")
    start_time_non_vec = time.time()
    theta_nv, tl_nv, vl_nv, ta_nv, va_nv = logistic_regression_gd_non_vectorized(X_train, y_train, X_val, y_val)
    end_time_non_vec = time.time()
    print(f"Trainingszeit (Nicht-Vektorisiert): {end_time_non_vec - start_time_non_vec:.2f} Sekunden")

    plot_learning_curves(tl_nv, vl_nv, 'Nicht-Vektorisiert: Verlust vs. Epochen', 'Verlust (Binary Cross-Entropy)',
                         'loss_non_vectorized_logistic')
    plot_learning_curves(ta_nv, va_nv, 'Nicht-Vektorisiert: Genauigkeit vs. Epochen', 'Genauigkeit',
                         'accuracy_non_vectorized_logistic')

    print("\nEvaluierung des nicht-vektorisierten Modells auf dem Testset:")
    evaluate_model_on_test_set(theta_nv, X_test, y_test)

    # --- Vektorisierte Implementierung ---
    print("\n--- Training Vektorisierte Logistische Regression ---")
    start_time_vec = time.time()
    theta_v, tl_v, vl_v, ta_v, va_v = logistic_regression_gd_vectorized(X_train, y_train, X_val, y_val)
    end_time_vec = time.time()
    print(f"Trainingszeit (Vektorisiert): {end_time_vec - start_time_vec:.2f} Sekunden")

    plot_learning_curves(tl_v, vl_v, 'Vektorisiert: Verlust vs. Epochen', 'Verlust (Binary Cross-Entropy)',
                         'loss_vectorized_logistic')
    plot_learning_curves(ta_v, va_v, 'Vektorisiert: Genauigkeit vs. Epochen', 'Genauigkeit',
                         'accuracy_vectorized_logistic')

    print("\nEvaluierung des vektorisierten Modells auf dem Testset:")
    # Wir können dieselbe Funktion verwenden, da das Theta der vektorisierten Version wahrscheinlich besser ist.
    # Wenn man beide Thetas getrennt evaluieren will, kann man die CM anders benennen.
    evaluate_model_on_test_set(theta_v, X_test, y_test)

    print(f"\nVergleich der Trainingszeiten:")
    print(f"  Nicht-Vektorisiert: {end_time_non_vec - start_time_non_vec:.2f}s")
    print(f"  Vektorisiert:       {end_time_vec - start_time_vec:.2f}s")

    print(
        f"\nAlle Ergebnisse und Plots der logistischen Regression wurden in '{cfg.results_dir_logistic}' gespeichert.")
    print("Logistische Regression Assignment Teil abgeschlossen.")