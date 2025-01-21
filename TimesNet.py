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
    # valid_segments = fhr_data[0:14400]
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
    zip_file_path = r'/sise/home/mayaroz/signals.zip'
    process_zip(zip_file_path)
    df = pd.read_csv('generate_data/timesnet_df.csv')

    #only fhr:
    df = df[['fhr']]

    # Define the sequence length and reshape
    sequence_length = 21620
    n_features = 1

    # Ensure the data is a multiple of sequence_length
    n_samples = len(df) // sequence_length
    reshaped_data = df['fhr'].values[:n_samples * sequence_length].reshape(n_samples, sequence_length, n_features)
    print(f"Reshaped data: {reshaped_data}")
    print(f"Reshaped data shape: {reshaped_data.shape}")


    # Split the reshaped data into train (70%), validation (15%), and test (15%)
    train_data, temp_data = train_test_split(reshaped_data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    print(f"Train shape: {train_data.shape}")
    print(f"Validation shape: {val_data.shape}")
    print(f"Test shape: {test_data.shape}")


    # Introduce missing values in validation and test sets
    val_data_with_missing = introduce_missing_values(val_data, rate=0.1,pattern="point")
    test_data_with_missing = introduce_missing_values(test_data, rate=0.1,pattern="point")

    # Prepare training, validation, and testing dictionaries
    train_set = {"X": train_data}
    val_set = {"X": val_data_with_missing, "X_ori": val_data}  # Include both original and missing data
    test_set = {"X": test_data_with_missing}

    # Mask for testing (optional for metric calculation)
    test_X_indicating_mask = np.isnan(test_data) ^ np.isnan(test_data_with_missing) #xor - true only for null that were added in masking (not initiall nulls)
    test_X_ori = np.nan_to_num(test_data)  # Replace NaNs with 0 for metrics



    # Initialize the model
    timesnet = TimesNet(
        n_steps=21620,                # Sequence length (number of timestamps per sample)
        n_features=1,                # Number of features (1 for FHR)
        n_layers=2,                  # Number of layers in the model
        top_k=1,                     # Top K imputation strategy
        d_model=128,                 # Dimension of the model (e.g., embedding size)
        d_ffn=512,                   # Dimension of the feedforward network
        n_kernels=5,                 # Number of kernels in the convolution layers
        dropout=0.5,                 # Dropout rate to prevent overfitting
        apply_nonstationary_norm=False, # Whether to apply non-stationary normalization
        batch_size=32,               # Batch size for training
        epochs=15,                   # Number of epochs for training
        patience=3,                  # Patience for early stopping
        optimizer=Adam(lr=1e-3),     # Optimizer (Adam with learning rate 0.001)
        num_workers=0,               # Number of workers for data loading
        device=None,                 # Device (None for automatic selection)
        saving_path="tutorial_results/imputation/timesnet", # Path to save results
        model_saving_strategy="best", # Save only the best model
    )

    # # Train the model
    timesnet.fit(train_set=train_set, val_set=val_set)

    # After training, evaluate the model on the test set
    timesnet_results = timesnet.predict(test_set)
    timesnet_imputation = timesnet_results["imputation"]

    # Masked MAE calculation (ignoring missing values)
    testing_mae = mean_absolute_error(test_X_ori[test_X_indicating_mask], timesnet_imputation[test_X_indicating_mask])
    print(f"Testing mean absolute error: {testing_mae:.4f}")
    testing_mse = mean_squared_error(test_X_ori[test_X_indicating_mask], timesnet_imputation[test_X_indicating_mask])
    print(f"Testing mean squared error: {testing_mse:.4f}")




    # # Load the TensorBoard log directory
    # log_dir  = "/sise/home/mayaroz/tutorial_results/imputation/timesnet/20250115_T163305/tensorboard/events.out.tfevents.1736951585.ise-6000-02.2211022.0.pypots"
    # # Initialize lists to store the loss values
    # train_loss_values = []
    # val_loss_values = []
    # steps = []

    # # Use TensorBoard to read the events files
    # event_file = log_dir
    # for event in tf.compat.v1.train.summary_iterator(event_file):
    #     print(event)
    #     for value in event.summary.value:
    #         print(1)

    #         if value.tag == 'train_loss':  # Tag for training loss
    #             step = event.step
    #             train_loss = value.simple_value
    #             train_loss_values.append(train_loss)
    #             steps.append(step)
    #         elif value.tag == 'val_loss':  # Tag for validation loss
    #             val_loss = value.simple_value
    #             val_loss_values.append(val_loss)

    # # Plot the training and validation losses
    # plt.figure(figsize=(10, 6))
    # plt.plot(steps, train_loss_values, label='Training Loss', color='blue')
    # plt.plot(steps, val_loss_values, label='Validation Loss', color='red')
    # plt.xlabel('Steps')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss over Time')
    # plt.legend()
    # plt.savefig(f'plots/training.jpg', format='jpg', dpi=300)  # Save as JPG








#####################example 
# set_random_seed()
# Load the dataset
# # Load the PhysioNet-2012 dataset
# physionet2012_dataset = benchpots.datasets.preprocess_physionet2012(subset="all", rate=0.1)
# # Take a look at the generated PhysioNet-2012 dataset, you'll find that everything has been prepared for you,
# # data splitting, normalization, additional artificially-missing values for evaluation, etc.
# print(physionet2012_dataset.keys())

# # assemble the datasets for training
# dataset_for_training = {
#     "X": physionet2012_dataset['train_X'],
# }
# print(dataset_for_training)
# # assemble the datasets for validation
# dataset_for_validating = {
#     "X": physionet2012_dataset['val_X'],
#     "X_ori": physionet2012_dataset['val_X_ori'],
# }
# # assemble the datasets for test
# dataset_for_testing = {
#     "X": physionet2012_dataset['test_X'],
# }
# ## calculate the mask to indicate the ground truth positions in test_X_ori, will be used by metric funcs to evaluate models
# test_X_indicating_mask = np.isnan(physionet2012_dataset['test_X_ori']) ^ np.isnan(physionet2012_dataset['test_X'])
# test_X_ori = np.nan_to_num(physionet2012_dataset['test_X_ori'])  # metric functions do not accpet input with NaNs, hence fill NaNs with 0

# # initialize the model
# timesnet = TimesNet(
#     n_steps=physionet2012_dataset['n_steps'],
#     n_features=physionet2012_dataset['n_features'],
#     n_layers=1,
#     top_k=1,
#     d_model=128,
#     d_ffn=512,
#     n_kernels=5,
#     dropout=0.5,
#     apply_nonstationary_norm=False,
#     batch_size=32,
#     # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
#     epochs=10,
#     # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
#     # You can leave it to defualt as None to disable early stopping.
#     patience=3,
#     # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
#     # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
#     optimizer=Adam(lr=1e-3),
#     # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
#     # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
#     # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
#     num_workers=0,
#     # just leave it to default as None, PyPOTS will automatically assign the best device for you.
#     # Set it as 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices, even parallelly on ['cuda:0', 'cuda:1']
#     device=None,
#     # set the path for saving tensorboard and trained model files
#     saving_path="tutorial_results/imputation/timesnet",
#     # only save the best model after training finished.
#     # You can also set it as "better" to save models performing better ever during training.
#     model_saving_strategy="best",
# )


# # train the model on the training set, and validate it on the validating set to select the best model for testing in the next step
# timesnet.fit(train_set=dataset_for_training, val_set=dataset_for_validating)

# # the testing stage, impute the originally-missing values and artificially-missing values in the test set
# timesnet_results = timesnet.predict(dataset_for_testing)
# timesnet_imputation = timesnet_results["imputation"]

# # calculate mean absolute error on the ground truth (artificially-missing values)
# testing_mae = calc_mae(
#     timesnet_imputation,
#     test_X_ori,
#     test_X_indicating_mask,
# )
# print(f"Testing mean absolute error: {testing_mae:.4f}")
