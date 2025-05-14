import os
import ast
import gc
import itertools
import random
import zipfile
import shutil
import csv
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from pathlib import Path

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler  # <-- added
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# pypots imports (make sure these libraries are installed and available)
from pypots.data import load_specific_dataset
from pypots.imputation import SAITS, TimesNet , PatchTST ,USGAN,MRNN,GRUD,CSDI,TimeMixer
from pypots.optim import Adam
from pypots.utils.metrics import calc_mae
from pypots.utils.random import set_random_seed

import benchpots
from pygrinder import block_missing, mcar, seq_missing

# print("CUDA available:", torch.cuda.is_available())
# print("Current device:", torch.cuda.current_device())
# print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", DEVICE)

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
                df_all.to_csv(f"generate_data/all_sliding_{name}.csv", index=False)

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
def evaluate_interpolation_baselines(test_masked, test_original, mask, window):
    """
    Now returns errors in the *normalized* scale (no inverse transform).
    """
    results = {}
    n_samples, n_steps, n_features = test_masked.shape

    methods = {
        "linear":    lambda s: s.interpolate(method="linear",   limit_direction="both"),
        "global_mean": lambda s: s.fillna(s.mean()),
        "global_median": lambda s: s.fillna(s.median()),
        "rolling_mean":      lambda s: s.fillna(s.rolling(window//10, min_periods=1).mean()).fillna(s.mean()),
        "rolling_median":    lambda s: s.fillna(s.rolling(window//10, min_periods=1).median()).fillna(s.median()),
        "LOCF / NOCB": lambda s: s.fillna(method="ffill").fillna(method="bfill"),
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
# Results 
# =============================================================================
def update_and_save_results(modelname, mse, mae, missing_rate, pattern, windowsize, stride, csv_path="imputation_results_summary.csv"):
    import pandas as pd
    import os

    result = {
        "modelname": modelname,
        "missing_rate": missing_rate,
        "pattern": pattern,
        "windowsize": windowsize,
        "stride": stride,
        "mse": mse,
        "mae": mae,
    }

    new_row = pd.DataFrame([result])

    # Check if file exists and is not empty
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        df = pd.read_csv(csv_path)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(csv_path, index=False)
    return df


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    zip_file_path = r'/sise/home/mayaroz/signals.zip'
    window_size = 1000
    stride = window_size // 2

    # 1) Build sliding-window CSVs
    process_zip_sliding_split(zip_file_path,
                              window_size=window_size,
                              stride=stride,
                              test_size=0.2,
                              val_size=0.5,
                              random_state=42)

    # 2) Load windows
    train_df = pd.read_csv('generate_data/all_sliding_train.csv')
    val_df   = pd.read_csv('generate_data/all_sliding_val.csv')
    test_df  = pd.read_csv('generate_data/all_sliding_test.csv')

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
    os.makedirs("saved_models", exist_ok=True)
    with open("saved_models/standard_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    train_data = train_scaled.reshape(n_train, n_steps, n_feats)
    val_data   = val_scaled.reshape(n_val,   n_steps, n_feats)
    test_data  = test_scaled.reshape(n_test,  n_steps, n_feats)
    # --- end normalization ---

    # 3) Introduce missing
    missing_rate = 0.18
    pattern = "subseq"
    train_miss = introduce_missing_values(train_data, rate=missing_rate, pattern=pattern)
    val_miss   = introduce_missing_values(val_data,   rate=missing_rate, pattern=pattern)
    test_miss  = introduce_missing_values(test_data,  rate=missing_rate, pattern=pattern)



    # Check how many values were truly masked (i.e., were not NaN originally, and became NaN in test_miss)
    original_not_nan = ~np.isnan(test_data)
    now_nan = np.isnan(test_miss)
    truly_masked = original_not_nan & now_nan
    total_values = np.prod(test_data.shape)  # total number of values, including those already missing
    masked_count = np.sum(truly_masked)
    true_missing_rate_all = masked_count / total_values
    print(f"→ missing rate for evaluation (out of all values): {true_missing_rate_all:.2%} ({masked_count} / {total_values})")


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

    # Define path for the CSV file where all results will be stored
    csv_path = Path("imputation_results_summary.csv")

    print("\nBaseline Imputation Results")
    for model_name, scores in baseline_results.items():
        mae = scores['mae']
        mse = scores['mse']
        print(f"{model_name:10s} → MAE: {mae:.4f}, MSE: {mse:.4f}")

        # Save the results to CSV
        update_and_save_results(
            modelname=model_name,
            mse=mse,
            mae=mae,
            missing_rate=missing_rate,
            pattern=pattern,  # or "point"/"block" if you change pattern
            windowsize=window_size,
            stride=stride,
            csv_path=str(csv_path)
        )


    d_model =16
    timesnet_model = TimesNet(
    n_steps=window_size,
    n_features=n_feats,
    n_layers=1,
    top_k=1,
    d_model=16,
    d_ffn=2 * d_model,
    n_kernels=5,
    dropout=0.2,
    apply_nonstationary_norm=False,
    batch_size=64,
    epochs=100,
    patience=5,
    optimizer=Adam(lr=0.001),
    num_workers=0,
    device=torch.device('cuda'),
    saving_path="gridsearch_results/timesnet",
    model_saving_strategy="best"
)

    timesnet_model.fit(train_set=train_set, val_set=val_set)
    res = timesnet_model.predict(test_set)
    imputed = res["imputation"]

    mse = mean_squared_error(test_X_ori[mask], imputed[mask])
    mae = mean_absolute_error(test_X_ori[mask], imputed[mask])
    print(f"→ TimesNet | MSE: {mse:.4f}, MAE: {mae:.4f}")
    # 4. Save to summary table
    update_and_save_results(
    modelname="TimesNet",
    mse=mse,
    mae=mae,
    missing_rate=missing_rate,
    pattern=pattern,
    windowsize=window_size,
    stride=stride
 )
    

    n_heads=4
    d_k = 16
    saits_model = SAITS(

    # Data Structure
    n_steps=window_size,
    n_features=1,

    # Model Architecture
    n_layers=1,
    d_ffn=d_model *2,

    # Attention Mechanism
    n_heads=4,
    d_k=16,
    d_model=n_heads*d_k,

    d_v=16,
    diagonal_attention_mask=True,

    # Training Objectives
    ORT_weight=1,
    MIT_weight=1,

    # Training Parameters
    batch_size=64,
    epochs=100,
    patience=5,
    dropout=0.1,
    attn_dropout=0.01,
    optimizer=Adam(lr=0.001),

    # Technical Parameters
    num_workers=4,
    device=None,
    saving_path="gridsearch_results/saits",
    model_saving_strategy="better",
    verbose=True
    )

    
    saits_model.fit(train_set=train_set, val_set=val_set)
    res = saits_model.predict(test_set)
    imputed = res["imputation"]

    mse = mean_squared_error(test_X_ori[mask], imputed[mask])
    mae = mean_absolute_error(test_X_ori[mask], imputed[mask])
    print(f"→ SAITS | MSE: {mse:.4f}, MAE: {mae:.4f}")
    # 4. Save to summary table
    update_and_save_results(
    modelname="SAITS",
    mse=mse,
    mae=mae,
    missing_rate=missing_rate,
    pattern=pattern,
    windowsize=window_size,
    stride=stride
 )


    csdi_model = CSDI(
        n_steps=n_steps,
        n_features=n_feats,
        n_layers=4,
        n_heads=4,
        n_channels=4,
        d_time_embedding=16,
        d_feature_embedding=8,
        d_diffusion_embedding=16,
        target_strategy="random",
        n_diffusion_steps=1,
        batch_size=64,
        epochs=100,
        patience=5,
        optimizer=Adam(lr=0.001),
        num_workers=0,
        device=None,
        saving_path="gridsearch_results/csdi",
        model_saving_strategy="best"
    )



    csdi_model.fit(train_set=train_set, val_set=val_set)
    res = csdi_model.predict(test_set)
    imputed = res["imputation"]
    #for csdi
    imputed = imputed.squeeze(axis=1)

    mse = mean_squared_error(test_X_ori[mask], imputed[mask])
    mae = mean_absolute_error(test_X_ori[mask], imputed[mask])
    print(f"→ CSDI | MSE: {mse:.4f}, MAE: {mae:.4f}")
    # 4. Save to summary table
    update_and_save_results(
    modelname="CSDI",
    mse=mse,
    mae=mae,
    missing_rate=missing_rate,
    pattern=pattern,
    windowsize=window_size,
    stride=stride
 )
    
    d_model = 16
    d_ffn_ratio=1
    timemixer_model = TimeMixer(
    n_steps=n_steps,
    n_features=n_feats,
    n_layers=1,
    top_k=4,
    d_model=d_model,
    d_ffn=d_ffn_ratio * d_model,
    dropout=0.3,
    epochs=100,
    patience=5,
    saving_path="gridsearch_results/TimeMixer",
    optimizer=Adam(lr=0.001),
    device=None,
)

    timemixer_model.fit(train_set=train_set, val_set=val_set)
    res = timemixer_model.predict(test_set)
    imputed = res["imputation"]

    mse = mean_squared_error(test_X_ori[mask], imputed[mask])
    mae = mean_absolute_error(test_X_ori[mask], imputed[mask])
    print(f"→ TIMEMIXER | MSE: {mse:.4f}, MAE: {mae:.4f}")
    # 4. Save to summary table
    update_and_save_results(
    modelname="TIMEMIXER",
    mse=mse,
    mae=mae,
    missing_rate=missing_rate,
    pattern=pattern,
    windowsize=window_size,
    stride=stride
 )
    



    # with open("saved_models/timemixer_model.pkl", "wb") as f:
    #     pickle.dump(timemixer_model, f)

    # grud_model =GRUD(
    #     n_steps=window_size,
    #     n_features=1,
    #     epochs=100,
    #     patience=5,
    #     rnn_hidden_size = config["rnn_hidden_size"],
    #     batch_size=config["batch_size"],
    #     optimizer=Adam(lr=config["lr"]),
    #     # Technical Parameters
    #     num_workers=4,
    #     device=None,
    #     saving_path="gridsearch_results/GRUD",
    #     model_saving_strategy="better",
    #     verbose=True
    # )


#     model.fit(train_set=train_set, val_set=val_set)
#     res = model.predict(test_set)
#     imputed = res["imputation"]

# # Compute MSE directly on the normalized imputed data
#     mse = mean_squared_error(
#         test_X_ori[mask],  # Ground truth (original data)
#         imputed[mask]     # Imputed values (in normalized scale)
#     )
#     print(f"→ MSE: {mse:.4f}")



#####################################################classification##################################################
