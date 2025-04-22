import os
import ast
import gc
import itertools
import random
import zipfile
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# pypots imports (make sure these libraries are installed and available)
from pypots.data import load_specific_dataset
from pypots.imputation import SAITS, TimesNet
from pypots.optim import Adam
from pypots.utils.metrics import calc_mae
from pypots.utils.random import set_random_seed

import benchpots
from pygrinder import block_missing, mcar, seq_missing

# =============================================================================
# Function to extract sliding windows from a single CSV file
# =============================================================================
def process_signal_sliding_window(file_path, window_size=2000, stride=2000//2):
    """
    Process a single CSV file to extract sliding windows.
    Each returned window is a tuple (window, file_id, start_index).
    """
    import numpy as np
    import pandas as pd

    # Read CSV file and extract the FHR column
    df = pd.read_csv(file_path)
    fhr_data = df["FHR"].values.astype(float)
    
    # Replace zeroes with NaN (if zero indicates missing data)
    fhr_data[fhr_data == 0] = np.nan
    
    file_id = os.path.basename(file_path)
    windows = []
    
    # Only process the file if its length is at least the window size
    if len(fhr_data) >= window_size:
        for start in range(0, len(fhr_data) - window_size + 1, stride):
            window = fhr_data[start:start + window_size]
            windows.append((window, file_id, start))
    else:
        print(f"Signal in {file_path} is too short for a window of size {window_size}. Skipping.")
    
    return windows

# =============================================================================
# Function to process the ZIP archive and split files into train/val/test sets
# =============================================================================
def process_zip_sliding_split(zip_path, window_size=100, stride=100//2,
                              test_size=0.2, val_size=0.5, random_state=42):
    """
    Process a ZIP file containing CSV files with FHR data.
    Splits the files by file_id into train, validation, and test groups first,
    then creates sliding windows for each group and saves them into separate CSV files.
    """
    results_train = []
    results_val = []
    results_test = []
    temp_dir = "extracted"  # Temporary extraction directory

    try:
        # Extract ZIP file into temporary directory
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_dir)
        
        # Assume files are inside a nested directory "signals/signals"
        nested_path = os.path.join(temp_dir, "signals", "signals")
        
        # Build a list of all CSV file paths
        all_files = []
        for root, _, files in os.walk(nested_path):
            for file in files:
                if file.endswith('.csv'):
                    all_files.append(os.path.join(root, file))
                    
        # Split file list into test and remaining files (train/validation)
        train_files, temp_files = train_test_split(
            all_files, test_size=test_size, random_state=random_state
        )
        # Further split train_val into training and validation sets
        val_files, test_files = train_test_split(
            temp_files, test_size= 0.5, random_state=random_state
        )

        print(f"Total files: {len(all_files)}")
        print(f"Training files: {len(train_files)}")
        print(f"Validation files: {len(val_files)}")
        print(f"Test files: {len(test_files)}")

        # Process training files: generate sliding windows and collect results
        for file_path in train_files:
            try:
                windows = process_signal_sliding_window(file_path, window_size, stride)
                for window, file_id, start_idx in windows:
                    df_window = pd.DataFrame({
                        "file_id": [file_id],
                        "start_index": [start_idx],
                        "fhr": [list(window)]  # Storing as a list for CSV
                    })
                    results_train.append(df_window)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Process validation files
        for file_path in val_files:
            try:
                windows = process_signal_sliding_window(file_path, window_size, stride)
                for window, file_id, start_idx in windows:
                    df_window = pd.DataFrame({
                        "file_id": [file_id],
                        "start_index": [start_idx],
                        "fhr": [list(window)]
                    })
                    results_val.append(df_window)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Process test files
        for file_path in test_files:
            try:
                windows = process_signal_sliding_window(file_path, window_size, stride)
                for window, file_id, start_idx in windows:
                    df_window = pd.DataFrame({
                        "file_id": [file_id],
                        "start_index": [start_idx],
                        "fhr": [list(window)]
                    })
                    results_test.append(df_window)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Save results into separate CSV files
        os.makedirs('generate_data', exist_ok=True)
        if results_train:
            train_df = pd.concat(results_train, ignore_index=True)
            print(f"Number of training windows: {len(train_df)}")
            train_df.to_csv(os.path.join('generate_data', 'timesnet_sliding_train.csv'), index=False)
        if results_val:
            val_df = pd.concat(results_val, ignore_index=True)
            print(f"Number of validation windows: {len(val_df)}")
            val_df.to_csv(os.path.join('generate_data', 'timesnet_sliding_val.csv'), index=False)
        if results_test:
            test_df = pd.concat(results_test, ignore_index=True)
            print(f"Number of test windows: {len(test_df)}")
            test_df.to_csv(os.path.join('generate_data', 'timesnet_sliding_test.csv'), index=False)
    
    finally:
        # Clean up the temporary extraction directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# =============================================================================
# Function to introduce missing values into data using a specified pattern.
# =============================================================================
def introduce_missing_values(X, rate, pattern, seq_len=8):
    """
    Introduces missing values in a 3D NumPy array.
    Supported patterns: 'point', 'subseq', and 'block'.
    """
    if not (0 <= rate <= 1):
        raise ValueError("Rate must be between 0 and 1")
    if X.ndim != 3:
        raise ValueError("Input X must be 3D [samples, timesteps, features]")

    X_missing = np.copy(X)
    n_samples, n_steps, n_features = X.shape

    if pattern == "point":
        return mcar(X, rate)
    
    elif pattern == "subseq":
        if n_steps < seq_len:
            raise ValueError(f"Timesteps ({n_steps}) < seq_len ({seq_len})")

        # Total possible starting positions per sample and feature
        possible_starts = n_steps - seq_len + 1
        total_positions = n_samples * n_features * possible_starts
        
        # Number of sequences to mask
        num_sequences = int(total_positions * rate / seq_len)
        
        if num_sequences > total_positions:
            raise ValueError(f"Rate {rate} requires {num_sequences} sequences but only {total_positions} available")

        # Build list of all valid (sample, feature, start) tuples
        all_positions = [(s, f, st) 
                         for s in range(n_samples)
                         for f in range(n_features)
                         for st in range(possible_starts)]
        
        # Randomly sample unique positions without replacement
        selected_positions = random.sample(all_positions, num_sequences)

        # Apply masking for each selected tuple
        for s, f, st in selected_positions:
            X_missing[s, st:st+seq_len, f] = np.nan

        return X_missing
    
    elif pattern == "block":
        return block_missing(X_missing, factor=rate)
    
    else:
        raise ValueError(f"Invalid pattern: {pattern}. Use 'point', 'subseq', or 'block'")


# =============================================================================
# Function to evaluate baseline imputation methods (e.g. linear, mean, median, locf)
# =============================================================================
def evaluate_interpolation_baselines(test_masked, test_original, mask):
    """
    Evaluate several simple interpolation methods.
    Returns a dictionary with MAE and MSE scores.
    """
    results = {}
    n_samples, n_steps, n_features = test_masked.shape

    methods = {
        "linear": lambda s: s.interpolate(method="linear", limit_direction="both"),
        "mean": lambda s: s.fillna(s.rolling(24, min_periods=1).mean()).fillna(s.mean()),
        "median": lambda s: s.fillna(s.rolling(24, min_periods=1).median()).fillna(s.median()),
        "locf_safe": lambda s: s.fillna(method="ffill").fillna(method="bfill"),
    }

    for name, func in methods.items():
        imputed = test_masked.copy()
        for i in range(n_samples):
            for f in range(n_features):
                series = pd.Series(imputed[i, :, f])
                # If all values are missing, fallback to zeros
                if series.isna().all():
                    imputed[i, :, f] = 0
                else:
                    imputed[i, :, f] = func(series).values
                
        valid_mask = mask & (~np.isnan(test_original))
        mae = mean_absolute_error(test_original[valid_mask], imputed[valid_mask])
        mse = mean_squared_error(test_original[valid_mask], imputed[valid_mask])
        results[name] = {"mae": mae, "mse": mse}

    return results

# =============================================================================
# Helper: Convert a string representation of a list into a NumPy array.
# =============================================================================
def convert_str_to_array(s):
    # Replace "nan" with Python None so ast.literal_eval can work
    s_fixed = s.replace("nan", "None")
    lst = ast.literal_eval(s_fixed)
    return np.array([np.nan if val is None else val for val in lst])

# =============================================================================
# Main pipeline
# =============================================================================
if __name__ == "__main__":
    # --------------------------
    # Step 1: Process the ZIP file and split by file_id
    # --------------------------
    # Change this path to the location of your ZIP file
    zip_file_path = r'/home/nicoleka/fetal_monitoring_final_project/signals.zip'
    
    # Define parameters for window extraction
    window_size = 75   # Adjust as needed
    stride = window_size // 2
    
    # Process the ZIP and create three CSV files for train/val/test
    process_zip_sliding_split(
        zip_file_path, window_size=window_size, stride=stride,
        test_size=0.2, val_size=0.5, random_state=42
    )
    
    # --------------------------
    # Step 2: Load CSV files and convert window data into NumPy arrays
    # --------------------------
    # Load each group from its separate CSV file
    train_df = pd.read_csv('generate_data/timesnet_sliding_train.csv')
    val_df = pd.read_csv('generate_data/timesnet_sliding_val.csv')
    test_df = pd.read_csv('generate_data/timesnet_sliding_test.csv')
    
    print("Training data sample:")
    print(train_df.head())
    print("Validation data sample:")
    print(val_df.head())
    print("Test data sample:")
    print(test_df.head())
    
    # Convert the string representation of the list in the 'fhr' column to arrays.
    train_windows = train_df['fhr'].apply(convert_str_to_array).tolist()
    val_windows = val_df['fhr'].apply(convert_str_to_array).tolist()
    test_windows = test_df['fhr'].apply(convert_str_to_array).tolist()
    
    # Reshape to add the feature dimension (assuming univariate data)
    n_features = 1
    train_data = np.array(train_windows).reshape(-1, window_size, n_features)
    val_data = np.array(val_windows).reshape(-1, window_size, n_features)
    test_data = np.array(test_windows).reshape(-1, window_size, n_features)
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # --------------------------
    # Step 3: Introduce missing values
    # --------------------------
    missing_rate = 0.01   # Adjust as desired
    train_data_with_missing = introduce_missing_values(train_data, rate=missing_rate, pattern="subseq")
    val_data_with_missing = introduce_missing_values(val_data, rate=missing_rate, pattern="subseq")
    test_data_with_missing = introduce_missing_values(test_data, rate=missing_rate, pattern="subseq")
    
    # Create data sets (here for model training and evaluation)
    train_set = {"X": train_data_with_missing, "X_ori": train_data}
    val_set = {"X": val_data_with_missing, "X_ori": val_data}
    test_set = {"X": test_data_with_missing}
    
    # Create a mask for evaluating imputation performance (if needed)
    test_X_indicating_mask = np.isnan(test_data) ^ np.isnan(test_data_with_missing)
    test_X_ori = np.nan_to_num(test_data)
    
    baseline_results = evaluate_interpolation_baselines(
        test_masked=test_data_with_missing,
        test_original=test_X_ori,
        mask=test_X_indicating_mask
    )
    
    print("\nBaseline Imputation Results:")
    for method, scores in baseline_results.items():
        print(method)
        print("MAE:", scores['mae'])
        print("MSE:", scores['mse'])
    
    # --------------------------
    # Step 4: Hyperparameter Grid Search and Model Training with TimesNet
    # --------------------------
    # Define grid search hyperparameters
    param_grid = {
        "batch_size": [16],
        "n_layers": [3],
        "top_k": [2],
        "d_model": [8, 16, 32, 64, 128, 256],
        # "d_ffn": [256],
        "n_kernels": [3],
        "dropout": [0.3],
        "lr": [0.0005]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = list(itertools.product(*values))
    
    best_mse = float("inf")
    best_config = None
    
    print(f"\nTotal hyperparameter combinations to evaluate: {len(combinations)}")
    
    for idx, combo in enumerate(combinations):
        # Clear CUDA cache if applicable
        torch.cuda.empty_cache()
        config = dict(zip(keys, combo))
        print(f"\n[{idx + 1}/{len(combinations)}] Trying config: {config}")
        try:
            d_model = config["d_model"]
            # Initialize the TimesNet model with the given configuration
            model = TimesNet(
                n_steps=window_size,
                n_features=n_features,
                n_layers=config["n_layers"],
                top_k=config["top_k"],
                d_model=d_model,
                d_ffn=d_model * 2,
                n_kernels=config["n_kernels"],
                dropout=config["dropout"],
                apply_nonstationary_norm=True,
                batch_size=config["batch_size"],
                epochs=100,
                patience=5,
                optimizer=Adam(lr=config["lr"], weight_decay=1e-4),
                num_workers=0,
                device=None,
                saving_path="gridsearch_results/timesnet",
                model_saving_strategy="best"
            )
    
            model.fit(train_set=train_set, val_set=val_set)
            results = model.predict(test_set)
            imputed = results["imputation"]
    
            mse = mean_squared_error(test_X_ori[test_X_indicating_mask], imputed[test_X_indicating_mask])
            print(f"→ MSE: {mse:.4f}")
    
            if mse < best_mse:
                best_mse = mse
                best_config = config
                print("✅ New best config!")
    
        except Exception as e:
            print(f"❌ Failed with config {config}: {e}")
    
    print("\n🏆 Best hyperparameter configuration:")
    print(best_config)
    print(f"Best MSE achieved: {best_mse:.4f}")
