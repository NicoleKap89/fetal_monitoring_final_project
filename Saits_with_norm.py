import os
import ast
import gc
import itertools
import random
import zipfile
import shutil
import csv

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

def load_and_normalize_files(train_paths, val_paths, test_paths):
    """
    Loads FHR data from CSV file paths, fits a StandardScaler on the training data,
    and returns normalized signals for train, val, and test sets.
    Each output list contains tuples of (normalized_signal_array, file_id).
    """

    def read_fhr(path):
        df = pd.read_csv(path)
        fhr = df["FHR"].values.astype(float)
        fhr[fhr == 0] = np.nan  # Treat 0 as missing
        return fhr

    # 1. Collect all valid FHR values from training files
    all_train_values = []
    for path in train_paths:
        fhr = read_fhr(path)
        all_train_values.extend(fhr[~np.isnan(fhr)])

    # 2. Fit the scaler on the training values only
    scaler = StandardScaler()
    scaler.fit(np.array(all_train_values).reshape(-1, 1))

    # 3. Helper function to normalize a list of files
    def normalize_files(file_paths):
        data_list = []
        for path in file_paths:
            fhr = read_fhr(path)
            mask = ~np.isnan(fhr)  # Only normalize valid values
            fhr_norm = fhr.copy()
            fhr_norm[mask] = scaler.transform(fhr[mask].reshape(-1, 1)).flatten()
            data_list.append((fhr_norm, os.path.basename(path)))
        return data_list

    # 4. Return normalized train/val/test lists of (data, file_id)
    return normalize_files(train_paths), normalize_files(val_paths), normalize_files(test_paths)


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

        train_files,val_files,test_files = load_and_normalize_files(train_files,val_files,test_files)


        for data_list, collector in [(train_files, results_train),
                              (val_files,   results_val),
                              (test_files,  results_test)]:
            for fhr_data, fid in data_list:
                try:
                    if len(fhr_data) >= window_size:
                        for start in range(0, len(fhr_data) - window_size + 1, stride):
                            window = fhr_data[start:start + window_size]
                            collector.append(pd.DataFrame({
                                "file_id":     [fid],
                                "start_index": [start],
                                "fhr":         [list(window)]
                            }))
                    else:
                        print(f"Signal {fid} too short. Skipping.")
                except Exception as e:
                    print(f"Error processing {fid}: {e}")


        os.makedirs('generate_data', exist_ok=True)
        for name, data in [("train", results_train),
                           ("val",   results_val),
                           ("test",  results_test)]:
            if data:
                df_all = pd.concat(data, ignore_index=True)
                print(f"Number of {name} windows: {len(df_all)}")
                df_all.to_csv(f"generate_data/saits_sliding_{name}.csv", index=False)

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


# def process_zip_sliding_split(zip_path, window_size=100, stride=100//2,
#                               test_size=0.2, val_size=0.5, random_state=42):
#     results_train, results_val, results_test = [], [], []
#     temp_dir = "extracted"

#     try:
#         with zipfile.ZipFile(zip_path, 'r') as z:
#             z.extractall(temp_dir)

#         nested = os.path.join(temp_dir, "signals", "signals")
#         all_files = [os.path.join(r, f)
#                      for r, _, files in os.walk(nested)
#                      for f in files if f.endswith('.csv')]

#         train_files, temp_files = train_test_split(all_files,
#                                                    test_size=test_size,
#                                                    random_state=random_state)
#         val_files, test_files = train_test_split(temp_files,
#                                                  test_size=0.5,
#                                                  random_state=random_state)

#         print(f"Total files: {len(all_files)}")
#         print(f"Training files: {len(train_files)}")
#         print(f"Validation files: {len(val_files)}")
#         print(f"Test files: {len(test_files)}")

#         for file_list, collector in [(train_files, results_train),
#                                      (val_files, results_val),
#                                      (test_files, results_test)]:
#             for fp in file_list:
#                 try:
#                     for window, fid, st in process_signal_sliding_window(fp, window_size, stride):
#                         collector.append(pd.DataFrame({
#                             "file_id":    [fid],
#                             "start_index":[st],
#                             "fhr":        [list(window)]
#                         }))
#                 except Exception as e:
#                     print(f"Error processing {fp}: {e}")

#         os.makedirs('generate_data', exist_ok=True)
#         for name, data in [("train", results_train),
#                            ("val",   results_val),
#                            ("test",  results_test)]:
#             if data:
#                 df_all = pd.concat(data, ignore_index=True)
#                 print(f"Number of {name} windows: {len(df_all)}")
#                 df_all.to_csv(f"generate_data/saits_sliding_{name}.csv", index=False)

#     finally:
#         if os.path.exists(temp_dir):
#             shutil.rmtree(temp_dir)

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
def evaluate_interpolation_baselines(test_masked, test_original, mask, window):
    """
    Now returns errors in the *normalized* scale (no inverse transform).
    """
    results = {}
    n_samples, n_steps, n_features = test_masked.shape

    methods = {
        "linear":    lambda s: s.interpolate(method="linear",   limit_direction="both"),
        "mean":      lambda s: s.fillna(s.rolling(window//10, min_periods=1).mean()).fillna(s.mean()),
        "median":    lambda s: s.fillna(s.rolling(window//10, min_periods=1).median()).fillna(s.median()),
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
    window_size = 50
    stride = window_size // 2

    # 1) Build sliding-window CSVs
    process_zip_sliding_split(zip_file_path,
                              window_size=window_size,
                              stride=stride,
                              test_size=0.2,
                              val_size=0.5,
                              random_state=42)

    # 2) Load windows
    train_df = pd.read_csv('generate_data/saits_sliding_train.csv')
    val_df   = pd.read_csv('generate_data/saits_sliding_val.csv')
    test_df  = pd.read_csv('generate_data/saits_sliding_test.csv')

    train_windows = train_df['fhr'].apply(convert_str_to_array).tolist()
    val_windows   = val_df['fhr'].apply(convert_str_to_array).tolist()
    test_windows  = test_df['fhr'].apply(convert_str_to_array).tolist()

    n_feats = 1
    train_data = np.array(train_windows).reshape(-1, window_size, n_feats)
    val_data   = np.array(val_windows).reshape(-1, window_size, n_feats)
    test_data  = np.array(test_windows).reshape(-1, window_size, n_feats)

    # --- normalize all sets ---
    # n_train, n_steps, _ = train_data.shape
    # n_val   = val_data.shape[0]
    # n_test  = test_data.shape[0]

    # train_flat = train_data.reshape(-1, n_feats)
    # val_flat   = val_data.reshape(-1, n_feats)
    # test_flat  = test_data.reshape(-1, n_feats)

    # scaler = StandardScaler()
    # train_scaled = scaler.fit_transform(train_flat)
    # val_scaled   = scaler.transform(val_flat)
    # test_scaled  = scaler.transform(test_flat)

    # train_data = train_scaled.reshape(n_train, n_steps, n_feats)
    # val_data   = val_scaled.reshape(n_val,   n_steps, n_feats)
    # test_data  = test_scaled.reshape(n_test,  n_steps, n_feats)
    # --- end normalization ---

    # 3) Introduce missing
    missing_rate = 0.18
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
        window=window_size 
    )

    print("\nBaseline Imputation Results (original scale):")
    for m, scores in baseline_results.items():
        print(f"{m:10s} → MAE: {scores['mae']:.4f}, MSE: {scores['mse']:.4f}")


    param_grid = {
        "n_layers": [1],
        "n_heads": [1],
        "d_k": [64],
        "d_v":[64],
        "d_ffn_ratio":[2], 
        "batch_size":[16], 
        "dropout": [0.2], 
        "attn_dropout":[0.2],
        "lr": [0.001]
    }

    keys, values = zip(*param_grid.items())
    combos = list(itertools.product(*values))

    best_mse, best_cfg = float("inf"), None
    print(f"\nTotal configs: {len(combos)}")


    file_path = "grid_search_results.csv"
    file_exists = os.path.isfile(file_path)
    mode = "a" if file_exists else "w"
    with open(file_path, mode, newline="") as csvfile:
        writer = csv.writer(csvfile)
        # only write header if brand-new file
        if not file_exists:
            writer.writerow(list(param_grid.keys()) + ["mse"])


        for idx, combo in enumerate(combos, 1):
            torch.cuda.empty_cache()
            config = dict(zip(keys, combo))
            print(f"\n[{idx}/{len(combos)}] Trying {config}")
                 
            try:
                n_heads = config["n_heads"]
                d_k = config["d_k"]
                d_model = n_heads * d_k
                model = SAITS(

                    # Data Structure
                    n_steps=window_size,
                    n_features=1,

                    # Model Architecture
                    n_layers=config["n_layers"],
                    d_model=d_model,
                    d_ffn=d_model * config["d_ffn_ratio"],

                    # Attention Mechanism
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=config["d_v"],
                    diagonal_attention_mask=True,

                    # Training Objectives
                    ORT_weight=1,
                    MIT_weight=1,

                    # Training Parameters
                    batch_size=config["batch_size"],
                    epochs=100,
                    patience=5,
                    dropout=config["dropout"],
                    attn_dropout=config["attn_dropout"],
                    optimizer=Adam(lr=config["lr"]),
                    
                    # Technical Parameters
                    num_workers=4,
                    device=None,
                    saving_path="gridsearch_results/saits",
                    model_saving_strategy="better",
                    verbose=True
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

                # ——— Write cfg + mse to CSV ———
                row = [config[k] for k in param_grid.keys()] + [mse]
                writer.writerow(row)

                if mse < best_mse:
                    best_mse, best_cfg = mse, config
                    print("✅ New best config!")

            except Exception as e:
                print(f"❌ Failed {config}: {e}")
        
        writer.writerow([])

    print("\n🏆 Best config:", best_cfg)
    print(f"Best MSE: {best_mse:.4f}")
