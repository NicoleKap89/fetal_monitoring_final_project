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
from sklearn.preprocessing import StandardScaler  # <-- added
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
    import numpy as np
    import pandas as pd

    df = pd.read_csv(file_path)
    fhr_data = df["FHR"].values.astype(float)
    fhr_data[fhr_data == 0] = np.nan

    file_id = os.path.basename(file_path)
    windows = []

    if len(fhr_data) >= window_size:
        for start in range(0, len(fhr_data) - window_size + 1, stride):
            window = fhr_data[start:start + window_size]
            windows.append((window, file_id, start))
    else:
        print(f"Signal in {file_path} is too short for a window of size {window_size}. Skipping.")

    return windows

# =============================================================================
# Splitting ZIP --> train/val/test CSVs
# =============================================================================
def process_zip_sliding_split(zip_path, window_size=100, stride=100//2,
                              test_size=0.2, val_size=0.5, random_state=42):
    results_train, results_val, results_test = [], [], []
    temp_dir = "extracted"

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_dir)

        nested = os.path.join(temp_dir, "signals", "signals")
        all_files = [os.path.join(r, f)
                     for r, _, files in os.walk(nested)
                     for f in files if f.endswith('.csv')]

        train_files, temp_files = train_test_split(all_files,
                                                   test_size=test_size,
                                                   random_state=random_state)
        val_files, test_files = train_test_split(temp_files,
                                                 test_size=0.5,
                                                 random_state=random_state)

        print(f"Total files: {len(all_files)}")
        print(f"Training files: {len(train_files)}")
        print(f"Validation files: {len(val_files)}")
        print(f"Test files: {len(test_files)}")

        for file_list, collector in [(train_files, results_train),
                                     (val_files, results_val),
                                     (test_files, results_test)]:
            for fp in file_list:
                try:
                    for window, fid, st in process_signal_sliding_window(fp, window_size, stride):
                        collector.append(pd.DataFrame({
                            "file_id":    [fid],
                            "start_index":[st],
                            "fhr":        [list(window)]
                        }))
                except Exception as e:
                    print(f"Error processing {fp}: {e}")

        os.makedirs('generate_data', exist_ok=True)
        for name, data in [("train", results_train),
                           ("val",   results_val),
                           ("test",  results_test)]:
            if data:
                df_all = pd.concat(data, ignore_index=True)
                print(f"Number of {name} windows: {len(df_all)}")
                df_all.to_csv(f"generate_data/timesnet_sliding_{name}.csv", index=False)

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

# =============================================================================
# Introduce missing values
# =============================================================================
def introduce_missing_values(X, rate, pattern, seq_len=8):
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
        possible = n_steps - seq_len + 1
        total_positions = n_samples * n_features * possible
        num_seq = int(total_positions * rate / seq_len)
        all_pos = [(s, f, st)
                   for s in range(n_samples)
                   for f in range(n_features)
                   for st in range(possible)]
        for s, f, st in random.sample(all_pos, num_seq):
            X_missing[s, st:st+seq_len, f] = np.nan
        return X_missing

    elif pattern == "block":
        return block_missing(X_missing, factor=rate)

    else:
        raise ValueError(f"Invalid pattern: {pattern}. Use 'point', 'subseq', or 'block'")

# =============================================================================
# Baseline interpolation + inverse-scaling evaluation
# =============================================================================
# =============================================================================
# 4) Baseline (original-scale errors) - Modified to keep errors in normalized scale
# =============================================================================
def evaluate_interpolation_baselines(test_masked, test_original, mask, scaler):
    """
    Now returns errors in the *normalized* scale (no inverse transform).
    """
    results = {}
    n_samples, n_steps, n_features = test_masked.shape

    methods = {
        "linear":    lambda s: s.interpolate(method="linear",   limit_direction="both"),
        "mean":      lambda s: s.fillna(s.rolling(24, min_periods=1).mean()).fillna(s.mean()),
        "median":    lambda s: s.fillna(s.rolling(24, min_periods=1).median()).fillna(s.median()),
        "locf_safe": lambda s: s.fillna(method="ffill").fillna(method="bfill"),
    }

    # Flatten original (normalized) to prepare for baseline error computation
    orig_flat_norm = test_original.reshape(-1, n_features)

    for name, func in methods.items():
        # 1) Impute on normalized data
        imputed_norm = test_masked.copy()
        for i in range(n_samples):
            for f in range(n_features):
                series = pd.Series(imputed_norm[i, :, f])
                if series.isna().all():
                    imputed_norm[i, :, f] = 0
                else:
                    imputed_norm[i, :, f] = func(series).values

        # 2) Flatten imputed values (already normalized)
        imp_flat_norm = imputed_norm.reshape(-1, n_features)

        # 3) Reshape back to original dimensions (keeping normalized scale)
        imp_norm = imp_flat_norm.reshape(n_samples, n_steps, n_features)

        # 4) Flatten the valid_mask to 2D to match the flattened arrays
        valid_mask_flat = mask.reshape(-1)

        # 5) Compute errors in the normalized scale
        mae = mean_absolute_error(orig_flat_norm[valid_mask_flat], imp_flat_norm[valid_mask_flat])
        mse = mean_squared_error(orig_flat_norm[valid_mask_flat], imp_flat_norm[valid_mask_flat])

        results[name] = {"mae": mae, "mse": mse}

    return results


# =============================================================================
# Helper: Convert string list to np.array
# =============================================================================
def convert_str_to_array(s):
    s_fixed = s.replace("nan", "None")
    lst = ast.literal_eval(s_fixed)
    return np.array([np.nan if v is None else v for v in lst])

# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    zip_file_path = r'/sise/home/mayaroz/signals.zip'
    window_size = 200
    stride = window_size // 2

    # 1) Build sliding-window CSVs
    process_zip_sliding_split(zip_file_path,
                              window_size=window_size,
                              stride=stride,
                              test_size=0.2,
                              val_size=0.5,
                              random_state=42)

    # 2) Load windows
    train_df = pd.read_csv('generate_data/timesnet_sliding_train.csv')
    val_df   = pd.read_csv('generate_data/timesnet_sliding_val.csv')
    test_df  = pd.read_csv('generate_data/timesnet_sliding_test.csv')

    train_windows = train_df['fhr'].apply(convert_str_to_array).tolist()
    val_windows   = val_df['fhr'].apply(convert_str_to_array).tolist()
    test_windows  = test_df['fhr'].apply(convert_str_to_array).tolist()

    n_feats = 1
    train_data = np.array(train_windows).reshape(-1, window_size, n_feats)
    val_data   = np.array(val_windows).reshape(-1, window_size, n_feats)
    test_data  = np.array(test_windows).reshape(-1, window_size, n_feats)

    # --- normalize all sets ---
    n_train, n_steps, _ = train_data.shape
    n_val   = val_data.shape[0]
    n_test  = test_data.shape[0]

    train_flat = train_data.reshape(-1, n_feats)
    val_flat   = val_data.reshape(-1, n_feats)
    test_flat  = test_data.reshape(-1, n_feats)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_flat)
    val_scaled   = scaler.transform(val_flat)
    test_scaled  = scaler.transform(test_flat)

    train_data = train_scaled.reshape(n_train, n_steps, n_feats)
    val_data   = val_scaled.reshape(n_val,   n_steps, n_feats)
    test_data  = test_scaled.reshape(n_test,  n_steps, n_feats)
    # --- end normalization ---

    # 3) Introduce missing
    missing_rate = 0.01
    train_miss = introduce_missing_values(train_data, rate=missing_rate, pattern="subseq")
    val_miss   = introduce_missing_values(val_data,   rate=missing_rate, pattern="subseq")
    test_miss  = introduce_missing_values(test_data,  rate=missing_rate, pattern="subseq")

    train_set = {"X": train_miss, "X_ori": train_data}
    val_set   = {"X": val_miss,   "X_ori": val_data}
    test_set  = {"X": test_miss}

    mask = np.isnan(test_data) ^ np.isnan(test_miss)
    test_X_ori = np.nan_to_num(test_data)

    # 4) Baseline (original-scale errors)
    baseline_results = evaluate_interpolation_baselines(
        test_masked=test_miss,
        test_original=test_X_ori,
        mask=mask,
        scaler=scaler
    )

    print("\nBaseline Imputation Results (original scale):")
    for m, scores in baseline_results.items():
        print(f"{m:10s} → MAE: {scores['mae']:.4f}, MSE: {scores['mse']:.4f}")

    # 5) Hyperparameter search + TimesNet
    param_grid = {
        "batch_size": [64],
        "n_layers":   [3],
        "top_k":      [2],
        "d_model":    [128],
        "d_ffn":      [256],
        "n_kernels":  [3],
        "dropout":    [0.3],
        "lr":         [0.0005]
    }
    keys, values = zip(*param_grid.items())
    combos = list(itertools.product(*values))

    best_mse, best_cfg = float("inf"), None
    print(f"\nTotal configs: {len(combos)}")
    for idx, combo in enumerate(combos, 1):
        torch.cuda.empty_cache()
        cfg = dict(zip(keys, combo))
        print(f"\n[{idx}/{len(combos)}] Trying {cfg}")
        try:
            model = TimesNet(
                n_steps=n_steps,
                n_features=n_feats,
                n_layers=cfg["n_layers"],
                top_k=cfg["top_k"],
                d_model=cfg["d_model"],
                d_ffn=cfg["d_ffn"],
                n_kernels=cfg["n_kernels"],
                dropout=cfg["dropout"],
                apply_nonstationary_norm=True,
                batch_size=cfg["batch_size"],
                epochs=100,
                patience=5,
                optimizer=Adam(lr=cfg["lr"], weight_decay=1e-4),
                num_workers=0,
                device=torch.device('cuda'),
                saving_path="gridsearch_results/timesnet",
                model_saving_strategy="best"
            )
            model.fit(train_set=train_set, val_set=val_set)
            res = model.predict(test_set)
            imputed = res["imputation"]

            # Keep the imputed data in the normalized scale (no inverse transform)
            # imp_flat = imputed.reshape(-1, n_feats)
            
            # Compute MSE directly on the normalized imputed data
            mse = mean_squared_error(
                test_X_ori[mask],  # Ground truth (original data)
                imputed[mask]     # Imputed values (in normalized scale)
            )
            print(f"→ MSE: {mse:.4f}")


            if mse < best_mse:
                best_mse, best_cfg = mse, cfg
                print("✅ New best config!")

        except Exception as e:
            print(f"❌ Failed {cfg}: {e}")

    print("\n🏆 Best config:", best_cfg)
    print(f"Best MSE: {best_mse:.4f}")


# # Standard library imports
# import ast
# import gc
# import itertools
# import os
# import random
# import zipfile

# # Third-party library imports
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import torch

# # scikit-learn imports
# from sklearn.impute import KNNImputer
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.model_selection import train_test_split

# # PyPOTS imports
# from pypots.data import load_specific_dataset
# from pypots.imputation import SAITS, TimesNet
# from pypots.optim import Adam
# from pypots.utils.metrics import calc_mae
# from pypots.utils.random import set_random_seed

# # Other specialized libraries
# import benchpots
# from pygrinder import block_missing, mcar, seq_missing


# def process_signal_sliding_window(file_path, window_size=2000, stride=2000//2):
#     """
#     Processes FHR signal data from a CSV file and extracts sliding windows of fixed length.
    
#     Args:
#         file_path (str): Path to the CSV file containing FHR data.
#         window_size (int): The fixed length for each window.
#         stride (int): Number of timesteps to shift between consecutive windows.
#                       A smaller stride means more overlap.
    
#     Returns:
#         list of tuples: Each tuple contains (window, file_id, start_index)
#                         where window is a numpy array of shape (window_size,).
#     """
#     import numpy as np
#     import pandas as pd
#     # Read CSV and extract FHR column
#     df = pd.read_csv(file_path)
#     fhr_data = df["FHR"].values.astype(float)
    
#     # Replace zeroes with NaN (if you want to treat 0 as missing data)
#     fhr_data[fhr_data == 0] = np.nan
    
#     file_id = os.path.basename(file_path)
#     windows = []
    
#     # Only process if the signal is long enough
#     if len(fhr_data) >= window_size:
#         for start in range(0, len(fhr_data) - window_size + 1, stride):
#             window = fhr_data[start:start + window_size]
#             windows.append((window, file_id, start))
#     else:
#         # Optionally, handle signals that are too short. You could choose to pad, discard, or skip.
#         print(f"Signal in {file_path} is too short for a window of size {window_size}. Skipping.")
    
#     return windows


# def process_zip_sliding(zip_path, window_size=2000, stride=2000//2):
#     """
#     Processes a ZIP file containing CSV files with FHR data and extracts sliding windows.
    
#     Args:
#         zip_path (str): Path to the ZIP file.
#         window_size (int): The fixed length for each extracted window.
#         stride (int): The sliding stride for the window.
    
#     Returns:
#         None. Saves a CSV file with sliding window data.
#     """
#     import os
#     import zipfile
#     import pandas as pd
#     results = []
#     temp_dir = "extracted"  # Temporary extraction directory

#     try:
#         with zipfile.ZipFile(zip_path, 'r') as z:
#             z.extractall(temp_dir)

#         nested_path = os.path.join(temp_dir, "signals", "signals")
        
#         # Iterate through the CSV files
#         for root, _, files in os.walk(nested_path):
#             for file in files:
#                 if file.endswith('.csv'):
#                     file_path = os.path.join(root, file)
#                     try:
#                         # Extract windows using the sliding window function
#                         windows = process_signal_sliding_window(file_path, window_size, stride)
#                         for window, file_id, start_idx in windows:
#                             # Save the window as a list. You can also consider storing as a npy file for efficiency.
#                             df_window = pd.DataFrame({
#                                 "file_id": [file_id],
#                                 "start_index": [start_idx],
#                                 "fhr": [list(window)]  # convert numpy array to list for CSV storage
#                             })
#                             # print(type(df_window['fhr']))
#                             # print(type(df_window['fhr'][0][0]))
#                             results.append(df_window)
#                     except Exception as e:
#                         print(f"Error processing {file_path}: {e}")

#         if results:
#             final_df = pd.concat(results, ignore_index=True)
#             unique_file_ids = final_df['file_id'].nunique()
#             print(f"Number of unique file_id values: {unique_file_ids}")
#             output_dir = 'generate_data'
#             os.makedirs(output_dir, exist_ok=True)
#             final_df.to_csv(os.path.join(output_dir, 'timesnet_sliding.csv'), index=False)
#         else:
#             print("No valid CSV files found or processed.")

#     finally:
#         # Clean up temporary directory
#         import shutil
#         if os.path.exists(temp_dir):
#             shutil.rmtree(temp_dir)


# def introduce_missing_values(X, rate, pattern, seq_len=8):
#     """
#     Introduces missing values into a NumPy array based on a specified pattern.
#     For 'subseq', applies the rate per sample and ensures unique sequences are masked.

#     Args:
#         X (np.ndarray): The input data array, expected shape [samples, timesteps, features].
#         rate (float): The approximate proportion of values to be masked (0.0 to 1.0).
#                       For 'subseq' and 'point', applied per sample.
#                       For 'block', acts as a 'factor' globally as per pygrinder.
#         pattern (str): The missingness pattern ('point', 'subseq', 'block').
#         seq_len (int): The fixed length of missing sequences for the 'subseq' pattern. Default is 8.

#     Returns:
#         np.ndarray: A copy of X with missing values introduced (represented as np.nan).
#     """
#     # Input validation
#     if not (0 <= rate <= 1):
#         raise ValueError("Rate must be between 0 and 1")
#     if X.ndim != 3:
#         raise ValueError("Input X must be 3D [samples, timesteps, features]")

#     X_missing = np.copy(X)
#     n_samples, n_steps, n_features = X.shape

#     if pattern == "point":
#         return mcar(X, rate)
    
#     elif pattern == "subseq":
#         if n_steps < seq_len:
#             raise ValueError(f"Timesteps ({n_steps}) < seq_len ({seq_len})")

#         # Pre-calculate all possible start positions
#         possible_starts = n_steps - seq_len + 1
#         total_positions = n_samples * n_features * possible_starts
        
#         # Calculate required sequences without per-sample loops
#         num_sequences = int(total_positions * rate / seq_len)
        
#         # Generate unique positions using random sampling
#         if num_sequences > total_positions:
#             raise ValueError(f"Rate {rate} requires {num_sequences} sequences but only {total_positions} available")

#         # Create all possible position tuples
#         all_positions = [(s, f, st) 
#                         for s in range(n_samples)
#                         for f in range(n_features)
#                         for st in range(possible_starts)]
        
#         # Randomly select without replacement
#         selected_positions = random.sample(all_positions, num_sequences)

#         # Apply masking in vectorized way
#         for s, f, st in selected_positions:
#             X_missing[s, st:st+seq_len, f] = np.nan

#         return X_missing
    
#     elif pattern == "block":
#         return block_missing(X_missing, factor=rate)
    
#     else:
#         raise ValueError(f"Invalid pattern: {pattern}. Use 'point', 'subseq', or 'block'")


# def downsample_by_averaging(data, window_size=4):
#     """
#     Downsample data by averaging every window_size timestamps.
    
#     Parameters:
#         data (np.ndarray): Input data with shape [samples, timesteps, features]
#         window_size (int): Number of timestamps to average together
    
#     Returns:
#         np.ndarray: Downsampled data with shape [samples, timesteps//window_size, features]
#     """
#     n_samples, n_steps, n_features = data.shape
    
#     # Calculate new sequence length after downsampling
#     new_length = n_steps // window_size
    
#     # Create output array
#     downsampled = np.zeros((n_samples, new_length, n_features))
    
#     # For each sample and feature, average the values in each window
#     for i in range(n_samples):
#         for j in range(n_features):
#             # Reshape to handle window_size groups
#             reshaped = data[i, :new_length*window_size, j].reshape(-1, window_size)
            
#             # Handle NaN values by using nanmean (mean ignoring NaNs)
#             downsampled[i, :, j] = np.nanmean(reshaped, axis=1)
    
#     return downsampled


# def evaluate_interpolation_baselines(test_masked, test_original, mask):
#     """
#     Evaluate multiple baseline interpolation methods on masked test data.

#     Parameters:
#         test_masked (np.ndarray): Test data with artificial missing values [samples, timesteps, features]
#         test_original (np.ndarray): Original complete test data (NaNs replaced with 0s for metric use)
#         mask (np.ndarray): Boolean mask indicating the artificially masked points (True = was masked)

#     Returns:
#         dict: MAE and MSE for each interpolation method
#     """
#     results = {}
#     n_samples, n_steps, n_features = test_masked.shape


#     methods = {
#         "linear": lambda s: s.interpolate(method="linear", limit_direction="both"),
#         # "cubic" : lambda s :s.interpolate(method='cubic', limit_direction='both', fill_value='extrapolate'),
#         "mean": lambda s: s.fillna(s.rolling(24, min_periods=1).mean()).fillna(s.mean()),
#         "median": lambda s: s.fillna(s.rolling(24, min_periods=1).median()).fillna(s.median()),
#         #"locf_safe" means fill forward (LOCF), then backfill any leading NaNs so that the sequence has no missing values at all.
#         "locf_safe": lambda s: s.fillna(method="ffill").fillna(method="bfill"),  # forward fill + backfill for safety

#     }

#     for name, func in methods.items():
#         imputed = test_masked.copy()
#         for i in range(n_samples):
#             for f in range(n_features):
#                 series = pd.Series(imputed[i, :, f])
#                 if series.isna().all():
#                     imputed[i, :, f] = 0  # Fallback for all-NaN
#                 else:
#                     imputed[i, :, f] = func(series).values
                
#         valid_mask = mask & (~np.isnan(test_original))  # Exclude original NaNs
#         mae = mean_absolute_error(test_original[valid_mask], imputed[valid_mask])
#         mse = mean_squared_error(test_original[valid_mask], imputed[valid_mask])
#         results[name] = {"mae": mae, "mse": mse}

#     return results


# import numpy as np

# def safe_eval(x):
#     # eval with a restricted globals dictionary that maps 'nan' to np.nan
#     return eval(x, {"__builtins__": None}, {"nan": np.nan})


# if __name__ == "__main__":
    
#     # os.environ['CUDA_LAUNCH_BLOCKING']="1"
#     # os.environ['TORCH_USE_CUDA_DSA'] = "1"
#     import torch
#     import numpy as np


#     # zip_file_path = r'/home/nicoleka/fetal_monitoring_final_project/signals.zip'
#     zip_file_path = r'/sise/home/mayaroz/signals.zip'
    
#     window_size = 100
#     stride = window_size//2
#     n_features = 1

#     # Process the ZIP file to generate sliding windows
#     process_zip_sliding(zip_file_path, window_size, stride)
    
#     # Load the CSV file with sliding window data
#     df = pd.read_csv('generate_data/timesnet_sliding.csv')
#     # print(df)
#     # print(df['fhr'][0][0])
#     # print(type(df['fhr'][0][0]))

#     df = df[['fhr']]
#     print(df)

#     # Convert each string representation into an actual NumPy array
#     # windows = df['fhr'].apply(lambda x: np.array((x))).tolist()

#     #print(windows)
#     # windows = df['fhr'].apply(lambda x: np.array(ast.literal_eval(x))).tolist()

#     import ast
#     import numpy as np

#     def convert_str_to_array(s):
#         # Replace nan with None (as a string) so ast.literal_eval can parse it.
#         s_fixed = s.replace("nan", "None")
#         # Convert string to list
#         lst = ast.literal_eval(s_fixed)
#         # Replace None with np.nan and return as a NumPy array
#         return np.array([np.nan if val is None else val for val in lst])

#     windows = df['fhr'].apply(convert_str_to_array).tolist()


#     # Reshape to add a feature dimension (e.g., 1 if it's a univariate series)
#     data_array = np.array(windows).reshape(-1, window_size, n_features)
    
#     print(f"Data array shape (sliding windows): {data_array.shape}")
    
#     # (Optional) Downsample or perform any other preprocessing here (using your downsample_by_averaging function)
#     # window_size_ds = 10  # for example
#     # downsampled_data = downsample_by_averaging(data_array, window_size=window_size_ds)
#     # print(f"Downsampled data shape: {downsampled_data.shape}")
    
#     # Continue with train/validation/test splits
#     from sklearn.model_selection import train_test_split
    
#     train_data, temp_data = train_test_split(data_array, test_size=0.2, random_state=42)
#     val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
#     print(f"Train shape: {train_data.shape}")
#     print(f"Validation shape: {val_data.shape}")
#     print(f"Test shape: {test_data.shape}")
    
#     # Introduce missing values as before
#     train_data_with_missing = introduce_missing_values(train_data, rate=0.01, pattern="subseq")
#     val_data_with_missing = introduce_missing_values(val_data, rate=0.01, pattern="subseq")
#     test_data_with_missing = introduce_missing_values(test_data, rate=0.01, pattern="subseq")
    
#     # Rest of your model training and evaluation code can remain largely the same
#     train_set = {"X": train_data_with_missing, "X_ori": train_data}
#     val_set = {"X": val_data_with_missing, "X_ori": val_data}
#     test_set = {"X": test_data_with_missing}
    
#     # Create a mask for evaluating imputation performance (if needed)
#     test_X_indicating_mask = np.isnan(test_data) ^ np.isnan(test_data_with_missing)
#     test_X_ori = np.nan_to_num(test_data)
    
#     baseline_results = evaluate_interpolation_baselines(
#         test_masked=test_data_with_missing,
#         test_original=test_X_ori,
#         mask=test_X_indicating_mask
#     )
    
#     for method, scores in baseline_results.items():
#         print(method)
#         print("MAE:", scores['mae'])
#         print("MSE:", scores['mse'])
    
#     # Grid Search over hyperparameters for your TimesNet model (unchanged)
#     import itertools
#     from pypots.imputation import TimesNet
#     from pypots.optim import Adam
#     from sklearn.metrics import mean_squared_error
    
#     param_grid = {
#         "batch_size": [32],
#         "n_layers": [1],
#         "top_k": [2],
#         "d_model": [128],
#         "d_ffn": [256],
#         "n_kernels": [3],
#         "dropout": [0.2],
#         "lr": [0.0001]
#     }
    
#     keys, values = zip(*param_grid.items())
#     combinations = list(itertools.product(*values))
    
#     best_mse = float("inf")
#     best_config = None
    
#     print(f"Total combinations to evaluate: {len(combinations)}")
    
#     for idx, combo in enumerate(combinations):
#         torch.cuda.empty_cache()
#         config = dict(zip(keys, combo))
#         print(f"\n[{idx + 1}/{len(combinations)}] Trying config: {config}")
#         try:
#             model = TimesNet(
#                 n_steps=window_size,
#                 n_features=n_features,
#                 n_layers=config["n_layers"],
#                 top_k=config["top_k"],
#                 d_model=config["d_model"],
#                 d_ffn=config["d_ffn"],
#                 n_kernels=config["n_kernels"],
#                 dropout=config["dropout"],
#                 apply_nonstationary_norm=True,
#                 batch_size=config["batch_size"],
#                 epochs=100,
#                 patience=5,
#                 optimizer=Adam(lr=config["lr"], weight_decay=1e-4),
#                 num_workers=0,
#                 device=None,
#                 saving_path="gridsearch_results/timesnet",
#                 model_saving_strategy="best"
#             )
    
#             model.fit(train_set=train_set, val_set=val_set)
#             results = model.predict(test_set)
#             imputed = results["imputation"]
    
#             mse = mean_squared_error(test_X_ori[test_X_indicating_mask], imputed[test_X_indicating_mask])
#             print(f"→ MSE: {mse:.4f}")
    
#             if mse < best_mse:
#                 best_mse = mse
#                 best_config = config
#                 print("✅ New best config!")
    
#         except Exception as e:
#             print(f"❌ Failed with config {config}: {e}")
    
#     print("\n🏆 Best config:")
#     print(best_config)
#     print(f"Best MSE: {best_mse:.4f}")