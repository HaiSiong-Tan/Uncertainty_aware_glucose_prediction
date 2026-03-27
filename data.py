import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os



def make_windows(X, Y, L=36, H=6):

    """Generate sliding input-output windows for time-series forecasting."""

    Xs, Ys = [], []
    T = len(X)

    for t in range(L, T - H + 1):
        Xs.append(X[t-L:t])
        Ys.append(Y[t:t+H])

    return np.array(Xs), np.array(Ys)



def load_single_patient(csv_path):

    """Load a patient CSV and return a time-series array of glucose and related features."""

    df = pd.read_csv(csv_path, sep=";", parse_dates=["time"])

    data_array = np.column_stack((
        df["glucose"].to_numpy(),
        df["heart_rate"].to_numpy(),
        df["calories"].to_numpy(),
        df["steps"].to_numpy(),
        df["basal_rate"].to_numpy(),
        df["bolus_volume_delivered"].to_numpy(),
        df["carb_input"].to_numpy()
    ))

    return data_array



def split_data(data_array):

    """Split time-series data into train, validation, and test sets."""

    N = data_array.shape[0]

    n_train = int(3/5 * N)
    n_val   = int(1/5 * N)

    train_data = data_array[:n_train]
    val_data   = data_array[n_train:n_train + n_val]
    test_data  = data_array[n_train + n_val:]

    train_val_data = np.concatenate((train_data, val_data), axis=0)

    return train_data, val_data, test_data, train_val_data



def extract_features(data, feature_names, column_map):

    """Extract specified features from time-series data."""

    indices = [column_map[f] for f in feature_names]
    X = data[:, indices]
    Y = data[:, column_map["glucose"]]  # target always glucose
    return X, Y



def prepare_dataset(data_dir, patient_ids, feature_names, column_map, L=36, H=1):

    """Prepare windowed train/val/test datasets from multiple patients and concatenate across patients."""

    train_input_list, train_output_list = [], []
    val_input_list, val_output_list = [], []
    test_input_list, test_output_list = [], []
    train_val_input_list, train_val_output_list = [], []

    for pid in patient_ids:
        csv_path = os.path.join(data_dir, f"HUPA00{pid}P.csv")

        data_array = load_single_patient(csv_path)

        train_data, val_data, test_data, train_val_data = split_data(data_array)

        X_train, Y_train = extract_features(train_data, feature_names, column_map)
        X_val, Y_val = extract_features(val_data, feature_names, column_map)
        X_test, Y_test = extract_features(test_data, feature_names, column_map)
        X_train_val, Y_train_val = extract_features(train_val_data, feature_names, column_map)

        train_input, train_output = make_windows(X_train, Y_train, L, H)
        val_input, val_output = make_windows(X_val, Y_val, L, H)
        test_input, test_output = make_windows(X_test, Y_test, L, H)
        train_val_input, train_val_output = make_windows(X_train_val, Y_train_val, L, H)

        train_input_list.append(train_input)
        train_output_list.append(train_output)
        val_input_list.append(val_input)
        val_output_list.append(val_output)
        test_input_list.append(test_input)
        test_output_list.append(test_output)
        train_val_input_list.append(train_val_input)
        train_val_output_list.append(train_val_output)

    # Concatenate across patients
    train_input = np.concatenate(train_input_list, axis=0)
    train_output = np.concatenate(train_output_list, axis=0)
    val_input = np.concatenate(val_input_list, axis=0)
    val_output = np.concatenate(val_output_list, axis=0)
    test_input = np.concatenate(test_input_list, axis=0)
    test_output = np.concatenate(test_output_list, axis=0)
    train_val_input = np.concatenate(train_val_input_list, axis=0)
    train_val_output = np.concatenate(train_val_output_list, axis=0)

    return {
        "train": (train_input, train_output),
        "val": (val_input, val_output),
        "test": (test_input, test_output),
        "train_val": (train_val_input, train_val_output),
    }


def compute_means_variances(data_dir, patient_ids, feature_names, column_map, L=36, H=1):

    """Computes means and variances for various input features using training dataset."""
    
    X_train_list = []
    for pid in patient_ids:
        csv_path = os.path.join(data_dir, f"HUPA00{pid}P.csv")

        data_array = load_single_patient(csv_path)

        train_data, val_data, test_data, train_val_data = split_data(data_array)

        X_train, Y_train = extract_features(train_data, feature_names, column_map)
        
        X_train_list.append(X_train)
    
    trainingset = np.concatenate(X_train_list, axis=0)

    mu_gen = np.mean(trainingset, axis=0)
    sigma_gen = np.std(trainingset, axis=0)
    mu_g = mu_gen[0]
    sigma_g = sigma_gen[0]

    return mu_g, sigma_g, mu_gen, sigma_gen



def normalize_features(X, mu, sigma, epsilon=1e-8):

    """Apply z-score normalization to features."""

    return (X - mu) / (sigma + epsilon)



class BGDataset(Dataset):
    """
    PyTorch Dataset for blood glucose prediction.
    """
    def __init__(self, inputs, outputs):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.outputs = torch.tensor(outputs, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]