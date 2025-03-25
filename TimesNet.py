import zipfile
import os
import numpy as np
import pandas as pd
from pypots.data import load_specific_dataset
from pypots.optim import Adam
from pypots.imputation import TimesNet
from pypots.utils.metrics import calc_mae
import numpy as np
import benchpots
from pypots.utils.random import set_random_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pygrinder import mcar, seq_missing, block_missing
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.pyplot as plt

#padding to max series length - 21,620 timestamps (90 min)
def process_signal(file_path):
    # Read CSV and extract the FHR column (or any other columns you need)
    df = pd.read_csv(file_path)
    # Assuming the FHR data is in a column named "FHR"
    fhr_data = df["FHR"].values
    length = len(fhr_data) #length of each file
    file_id = os.path.basename(file_path)  # Extract the file name or any other identifier

    if length > 21620:
        valid_segments = fhr_data[:21620]
    else:
    # Pad the signal with NaNs or zeros if it's shorter than 21,620 timestamps
        padding = np.full((21620 - length,), np.nan)  # Use np.nan for missing values or np.zeros for zero padding
    valid_segments = np.concatenate([fhr_data, padding])
    # Replace 0 values in the data with NaN for further processing
    valid_segments = np.where(valid_segments == 0, np.nan, valid_segments)

    return valid_segments, file_id , length




def process_zip(zip_path):
    results = []
    temp_dir = "extracted"  # Temporary directory to extract files
    minimum_len = np.inf
    maximum_len = -np.inf


    with zipfile.ZipFile(zip_path, 'r') as z:
        # Extract all files to the temporary directory
        z.extractall(temp_dir)

        # Path inside the ZIP where CSV files are stored
        nested_path = os.path.join(temp_dir, "signals", "signals")

        # Iterate through all CSV files in the nested directory
        for root, _, files in os.walk(nested_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    try:
                        # Call the process_signal function to extract valid segments
                        valid_segments, file_id,length = process_signal(file_path)
                        if length <= minimum_len:
                            minimum_len = length
                        if length >= maximum_len:
                            maximum_len = length
                        # Loop through the valid segments and extract the desired information
                        # for segment in valid_segments:
                        result = pd.DataFrame({
                            "file_id": file_id ,
                            "fhr": valid_segments
                         })
                        results.append(result)

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    # Concatenate all the results into one DataFrame
    final_df = pd.concat(results, ignore_index=True)
    # Count the number of unique 'fileid' values in the final_df
    unique_file_ids = final_df['file_id'].nunique()
    # Print the result
    print(f"Number of unique file_id values: {unique_file_ids}")
    os.makedirs('generate_data', exist_ok=True)
    final_df.to_csv(f'generate_data/timesnet_df.csv', index=False)

    
    # Clean up temporary directory
    for root, _, files in os.walk(temp_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        os.rmdir(root)
    print(f'max len file: {maximum_len}') # 21620
    return final_df



def introduce_missing_values(X, rate, pattern):
    if pattern == "point":
        return mcar(X, rate)
    elif pattern == "subseq":
        return seq_missing(X, rate)
    elif pattern == "block":
        return block_missing(X, factor=rate)
    else:
        raise ValueError(f"Unknown missingness pattern: {pattern}")


#padding
#normalization 
# Main block to execute
if __name__ == "__main__":
    zip_file_path = r'/home/nicoleka/fetal_monitoring_final_project/signals.zip'
    process_zip(zip_file_path)
    df = pd.read_csv('generate_data/timesnet_df.csv')

    #only fhr:
    df = df[['fhr']]

    # Define the sequence length and reshape
    sequence_length = 21620
    n_features = 1

    # Ensure the data is a multiple of sequence_length
    n_samples = len(df) // sequence_length

    print(n_samples)

    reshaped_data = df['fhr'].values[:n_samples * sequence_length].reshape(n_samples, sequence_length, n_features)
    print(f"Reshaped data: {reshaped_data}")
    print(f"Reshaped data shape: {reshaped_data.shape}")


    # Split the reshaped data into train (70%), validation (15%), and test (15%)
    train_data, temp_data = train_test_split(reshaped_data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print(f"Train shape: {train_data.shape}")
    print(f"Validation shape: {val_data.shape}")
    print(f"Test shape: {test_data.shape}")

    #new
    train_data_with_missing = introduce_missing_values(train_data, rate=0.1,pattern="point")

    # Introduce missing values in validation and test sets
    val_data_with_missing = introduce_missing_values(val_data, rate=0.1,pattern="point")
    test_data_with_missing = introduce_missing_values(test_data, rate=0.1,pattern="point")

    # Prepare training, validation, and testing dictionaries
    # train_set = {"X": train_data}

    train_set = {"X": train_data_with_missing, "X_ori" : train_data}
    val_set = {"X": val_data_with_missing, "X_ori": val_data}  # Include both original and missing data
    test_set = {"X": test_data_with_missing}

    # Mask for testing (optional for metric calculation)
    test_X_indicating_mask = np.isnan(test_data) ^ np.isnan(test_data_with_missing) #xor - true only for null that were added in masking (not initiall nulls)
    test_X_ori = np.nan_to_num(test_data)  # Replace NaNs with 0 for metrics



    # # Initialize the model
    # timesnet = TimesNet(
    #     n_steps=21620,                # Sequence length (number of timestamps per sample)
    #     n_features=1,                # Number of features (1 for FHR)
    #     n_layers=2,                  # Number of layers in the model
    #     top_k=1,                     # Top K imputation strategy
    #     d_model=128,                 # Dimension of the model (e.g., embedding size)
    #     d_ffn=512,                   # Dimension of the feedforward network
    #     n_kernels=5,                 # Number of kernels in the convolution layers
    #     dropout=0.3,                 # Dropout rate to prevent overfitting
    #     apply_nonstationary_norm=True, # Whether to apply non-stationary normalization ???
    #     batch_size=32,               # Batch size for training
    #     epochs=10,                   # Number of epochs for training
    #     patience=3,                  # Patience for early stopping
    #     optimizer=Adam(lr=1e-3),     # Optimizer (Adam with learning rate 0.001)
    #     num_workers=0,               # Number of workers for data loading
    #     device=None,                 # Device (None for automatic selection)
    #     saving_path="tutorial_results/imputation/timesnet", # Path to save results
    #     model_saving_strategy="best", # Save only the best model
    # )

    # # # Train the model
    # timesnet.fit(train_set=train_set, val_set=val_set)

    # # After training, evaluate the model on the test set
    # timesnet_results = timesnet.predict(test_set)
    # timesnet_imputation = timesnet_results["imputation"]

    # # Masked MAE calculation (ignoring missing values)
    # testing_mae = mean_absolute_error(test_X_ori[test_X_indicating_mask], timesnet_imputation[test_X_indicating_mask])
    # print(f"Testing mean absolute error: {testing_mae:.4f}")
    # testing_mse = mean_squared_error(test_X_ori[test_X_indicating_mask], timesnet_imputation[test_X_indicating_mask])
    # print(f"Testing mean squared error: {testing_mse:.4f}")



import itertools
from pypots.imputation import TimesNet
from pypots.optim import Adam
from sklearn.metrics import mean_squared_error

# Your hyperparameter grid
param_grid = {
    "n_layers": [1, 2, 3],
    "top_k": [1, 2, 3, 4, 5],
    "d_model": [128, 256, 512, 1024],
    "d_ffn": [128, 256, 512, 1024],
    "n_kernels": [4, 5, 6],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "lr": [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]  # expanded for manual grid
}

# Create all combinations
keys, values = zip(*param_grid.items())
combinations = list(itertools.product(*values))

best_mse = float("inf")
best_config = None

print(f"Total combinations to evaluate: {len(combinations)}")

for idx, combo in enumerate(combinations):
    config = dict(zip(keys, combo))
    print(f"\n[{idx + 1}/{len(combinations)}] Trying config: {config}")

    try:
        model = TimesNet(
            n_steps=21620,
            n_features=1,
            n_layers=config["n_layers"],
            top_k=config["top_k"],
            d_model=config["d_model"],
            d_ffn=config["d_ffn"],
            n_kernels=config["n_kernels"],
            dropout=config["dropout"],
            apply_nonstationary_norm=True,
            batch_size=32,
            epochs=30,  # shorter training for tuning
            patience=3,
            optimizer=Adam(lr=config["lr"]),
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

print("\n🏆 Best config:")
print(best_config)
print(f"Best MSE: {best_mse:.4f}")
