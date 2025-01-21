import numpy as np
import pandas as pd
import zipfile
import os
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_squared_error
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from tsfresh import extract_features
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import feature_calculators
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import inspect
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pyarrow
import dask.dataframe as dd
import fastparquet
from tsfresh.utilities.distribution import MultiprocessingDistributor
from tsfresh.feature_extraction import feature_calculators
from tsfresh import select_features
from tabulate import tabulate
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import CubicHermiteSpline
from xgboost import XGBClassifier



#create lables dictionary:
tabular_data = pd.read_csv(r'/sise/home/mayaroz/ann_db.csv')
data = tabular_data.T
data.columns = data.iloc[0]
data = data[1:]
ph_dict = data['pH'].to_dict()
#1 is the abnormal class (ph < 7.05)
#0 is the normal class (ph >= 7.05)
class_dict = {key: (0 if float(value) > 7.05 else 1) for key, value in ph_dict.items()} #as in boost ensemble paper
class_dict = {int(key): value for key, value in class_dict.items()}
# Count of keys with value = 0
count_0 = sum(1 for value in class_dict.values() if value == 0)
# Count of keys with value = 1
count_1 = sum(1 for value in class_dict.values() if value == 1)





def calculate_derivatives(timestamps, values):
    # Calculate central difference derivatives for the inner points
    dxdy = np.zeros_like(values)
    
    # Use central differences for the middle points
    for i in range(1, len(timestamps) - 1):
        dxdy[i] = (values[i + 1] - values[i - 1]) / (timestamps[i + 1] - timestamps[i - 1])
    
    # Use forward difference for the first point
    dxdy[0] = (values[1] - values[0]) / (timestamps[1] - timestamps[0])
    
    # Use backward difference for the last point
    dxdy[-1] = (values[-1] - values[-2]) / (timestamps[-1] - timestamps[-2])
    
    return dxdy



def test_imputation(series):
    # Forward fill - propagate the last valid value forward
    series_filled = series.ffill()
    # Apply backward fill - propagate the next valid value backward
    series_filled = series_filled.bfill()
    # series_filled = series.interpolate(method='linear', limit_direction='both')
    return series_filled


def process_signal(file_path, interpolation_type="linear", fill_method="interpolation"):
    data = pd.read_csv(file_path)
    data['FHR'] = data['FHR'].replace(0, np.nan)
    
    if fill_method == "interpolation":
        if interpolation_type == "linear":
            data['FHR'] = data['FHR'].interpolate(method='linear', limit_direction='both')
        elif interpolation_type == "cubic":
            data['FHR'] = data['FHR'].interpolate(method='cubic', limit_direction='both', fill_value='extrapolate')
        elif interpolation_type == "hermite":
            known_indices = ~data['FHR'].isna()
            known_timestamps = data.loc[known_indices, 'seconds'].values
            known_values = data.loc[known_indices, 'FHR'].values
            missing_indices = data['FHR'].isna()
            missing_timestamps = data.loc[missing_indices, 'seconds'].values
            known_derivatives = calculate_derivatives(known_timestamps, known_values)
            hermite_spline = CubicHermiteSpline(known_timestamps, known_values, known_derivatives,extrapolate='periodic')
            hermite_interpolated = hermite_spline(missing_timestamps)
            data.loc[missing_indices, 'FHR'] = hermite_interpolated

    elif fill_method == "test imputation":
        data['FHR'] = test_imputation(data['FHR'])
    return data

def prepare_data_for_tsfresh(df, file_id):
    tsfresh_df = df[['seconds', 'FHR']].rename(columns={'seconds': 'time', 'FHR': 'value'})
    tsfresh_df['id'] = file_id

    return tsfresh_df

def process_zip(zip_path, interpolation_type="linear"):
    temp_dir = "temp_extracted_data"
    train_data, test_data = [], []

    # Extract files from zip
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(temp_dir)
        nested_path = os.path.join(temp_dir, "signals", "signals")
        all_files = [os.path.join(root, file)
                     for root, _, files in os.walk(nested_path)
                     for file in files if file.endswith('.csv')]
        
        # Split files into train and test sets
        train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
        
        # Process training files
        for idx, file in enumerate(train_files):
            file_name = os.path.basename(file)
            file_id = int(file_name.split(".csv")[0])
            try:
                processed_data = process_signal(file, interpolation_type=interpolation_type, fill_method="interpolation")
                train_data.append(prepare_data_for_tsfresh(processed_data, file_id=file_id))
            except Exception as e:
                print(f"Error processing train file {file}: {e}")
        
        # Process testing files
        for idx, file in enumerate(test_files):
            file_name = os.path.basename(file)
            file_id = int(file_name.split(".csv")[0])
            try:
                processed_data = process_signal(file, fill_method="test imputation")
                test_data.append(prepare_data_for_tsfresh(processed_data, file_id=file_id))
            except Exception as e:
                print(f"Error processing test file {file}: {e}")
    
    # Clean up temporary directory
    for root, _, files in os.walk(temp_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        os.rmdir(root)

    return train_data, test_data


def evaluate_models(zip_file_path, class_dict, interpolation_types=["linear","hermite"]):
    results_all = []  # To store results for both interpolation methods
    for interpolation_type in interpolation_types:
        print(f"\nEvaluating with {interpolation_type} interpolation:")
        
        # Process data with the given interpolation method
        train_data, test_data = process_zip(zip_file_path, interpolation_type)
        train_df = pd.concat(train_data, axis=0)
        test_df = pd.concat(test_data, axis=0)

        
        # Align labels
        class_df = pd.DataFrame(list(class_dict.items()), columns=['id', 'label']).sort_values(by='id')
        train_labels = class_df[class_df['id'].isin(train_df['id'])].set_index('id')['label']
        test_labels = class_df[class_df['id'].isin(test_df['id'])].set_index('id')['label']

        default_fc_parameters1 = {
            "mean":None,  # Baseline (mean)
            "standard_deviation": None,  # Standard deviation
            "root_mean_square" : None,
            "variance" : None,
            "mean_abs_change": None,
            'autocorrelation': [{'lag': 0}, {'lag': 1}, {'lag': 2}, {'lag': 3}, {'lag': 4}, {'lag': 5}, {'lag': 6}, {'lag': 7}, {'lag': 8}, {'lag': 9}],  # Lagged autocorrelation (ACO)
            'fft_coefficient': [{'coeff': 0, 'attr': 'real'}, {'coeff': 1, 'attr': 'real'}, {'coeff': 2, 'attr': 'real'}, {'coeff': 3, 'attr': 'real'}, {'coeff': 4, 'attr': 'real'}, {'coeff': 5, 'attr': 'real'}, {'coeff': 6, 'attr': 'real'}, {'coeff': 7, 'attr': 'real'}, {'coeff': 8, 'attr': 'real'}, {'coeff': 9, 'attr': 'real'}, {'coeff': 10, 'attr': 'real'}, {'coeff': 11, 'attr': 'real'}, {'coeff': 12, 'attr': 'real'}, {'coeff': 13, 'attr': 'real'}, {'coeff': 14, 'attr': 'real'}, {'coeff': 15, 'attr': 'real'}, {'coeff': 16, 'attr': 'real'}, {'coeff': 17, 'attr': 'real'}, {'coeff': 18, 'attr': 'real'}, {'coeff': 19, 'attr': 'real'}, {'coeff': 20, 'attr': 'real'}, {'coeff': 21, 'attr': 'real'}, {'coeff': 22, 'attr': 'real'}, {'coeff': 23, 'attr': 'real'}, {'coeff': 24, 'attr': 'real'}, {'coeff': 25, 'attr': 'real'}, {'coeff': 26, 'attr': 'real'}, {'coeff': 27, 'attr': 'real'}, {'coeff': 28, 'attr': 'real'}, {'coeff': 29, 'attr': 'real'}, {'coeff': 30, 'attr': 'real'}, {'coeff': 31, 'attr': 'real'}, {'coeff': 32, 'attr': 'real'}, {'coeff': 33, 'attr': 'real'}, {'coeff': 34, 'attr': 'real'}, {'coeff': 35, 'attr': 'real'}, {'coeff': 36, 'attr': 'real'}, {'coeff': 37, 'attr': 'real'}, {'coeff': 38, 'attr': 'real'}, {'coeff': 39, 'attr': 'real'}, {'coeff': 40, 'attr': 'real'}, {'coeff': 41, 'attr': 'real'}, {'coeff': 42, 'attr': 'real'}, {'coeff': 43, 'attr': 'real'}, {'coeff': 44, 'attr': 'real'}, {'coeff': 45, 'attr': 'real'}, {'coeff': 46, 'attr': 'real'}, {'coeff': 47, 'attr': 'real'}, {'coeff': 48, 'attr': 'real'}, {'coeff': 49, 'attr': 'real'}, {'coeff': 50, 'attr': 'real'}, {'coeff': 51, 'attr': 'real'}, {'coeff': 52, 'attr': 'real'}, {'coeff': 53, 'attr': 'real'}, {'coeff': 54, 'attr': 'real'}, {'coeff': 55, 'attr': 'real'}, {'coeff': 56, 'attr': 'real'}, {'coeff': 57, 'attr': 'real'}, {'coeff': 58, 'attr': 'real'}, {'coeff': 59, 'attr': 'real'}, {'coeff': 60, 'attr': 'real'}, {'coeff': 61, 'attr': 'real'}, {'coeff': 62, 'attr': 'real'}, {'coeff': 63, 'attr': 'real'}, {'coeff': 64, 'attr': 'real'}, {'coeff': 65, 'attr': 'real'}, {'coeff': 66, 'attr': 'real'}, {'coeff': 67, 'attr': 'real'}, {'coeff': 68, 'attr': 'real'}, {'coeff': 69, 'attr': 'real'}, {'coeff': 70, 'attr': 'real'}, {'coeff': 71, 'attr': 'real'}, {'coeff': 72, 'attr': 'real'}, {'coeff': 73, 'attr': 'real'}, {'coeff': 74, 'attr': 'real'}, {'coeff': 75, 'attr': 'real'}, {'coeff': 76, 'attr': 'real'}, {'coeff': 77, 'attr': 'real'}, {'coeff': 78, 'attr': 'real'}, {'coeff': 79, 'attr': 'real'}, {'coeff': 80, 'attr': 'real'}, {'coeff': 81, 'attr': 'real'}, {'coeff': 82, 'attr': 'real'}, {'coeff': 83, 'attr': 'real'}, {'coeff': 84, 'attr': 'real'}, {'coeff': 85, 'attr': 'real'}, {'coeff': 86, 'attr': 'real'}, {'coeff': 87, 'attr': 'real'}, {'coeff': 88, 'attr': 'real'}, {'coeff': 89, 'attr': 'real'}, {'coeff': 90, 'attr': 'real'}, {'coeff': 91, 'attr': 'real'}, {'coeff': 92, 'attr': 'real'}, {'coeff': 93, 'attr': 'real'}, {'coeff': 94, 'attr': 'real'}, {'coeff': 95, 'attr': 'real'}, {'coeff': 96, 'attr': 'real'}, {'coeff': 97, 'attr': 'real'}, {'coeff': 98, 'attr': 'real'}, {'coeff': 99, 'attr': 'real'}, {'coeff': 0, 'attr': 'imag'}, {'coeff': 1, 'attr': 'imag'}, {'coeff': 2, 'attr': 'imag'}, {'coeff': 3, 'attr': 'imag'}, {'coeff': 4, 'attr': 'imag'}, {'coeff': 5, 'attr': 'imag'}, {'coeff': 6, 'attr': 'imag'}, {'coeff': 7, 'attr': 'imag'}, {'coeff': 8, 'attr': 'imag'}, {'coeff': 9, 'attr': 'imag'}, {'coeff': 10, 'attr': 'imag'}, {'coeff': 11, 'attr': 'imag'}, {'coeff': 12, 'attr': 'imag'}, {'coeff': 13, 'attr': 'imag'}, {'coeff': 14, 'attr': 'imag'}, {'coeff': 15, 'attr': 'imag'}, {'coeff': 16, 'attr': 'imag'}, {'coeff': 17, 'attr': 'imag'}, {'coeff': 18, 'attr': 'imag'}, {'coeff': 19, 'attr': 'imag'}, {'coeff': 20, 'attr': 'imag'}, {'coeff': 21, 'attr': 'imag'}, {'coeff': 22, 'attr': 'imag'}, {'coeff': 23, 'attr': 'imag'}, {'coeff': 24, 'attr': 'imag'}, {'coeff': 25, 'attr': 'imag'}, {'coeff': 26, 'attr': 'imag'}, {'coeff': 27, 'attr': 'imag'}, {'coeff': 28, 'attr': 'imag'}, {'coeff': 29, 'attr': 'imag'}, {'coeff': 30, 'attr': 'imag'}, {'coeff': 31, 'attr': 'imag'}, {'coeff': 32, 'attr': 'imag'}, {'coeff': 33, 'attr': 'imag'}, {'coeff': 34, 'attr': 'imag'}, {'coeff': 35, 'attr': 'imag'}, {'coeff': 36, 'attr': 'imag'}, {'coeff': 37, 'attr': 'imag'}, {'coeff': 38, 'attr': 'imag'}, {'coeff': 39, 'attr': 'imag'}, {'coeff': 40, 'attr': 'imag'}, {'coeff': 41, 'attr': 'imag'}, {'coeff': 42, 'attr': 'imag'}, {'coeff': 43, 'attr': 'imag'}, {'coeff': 44, 'attr': 'imag'}, {'coeff': 45, 'attr': 'imag'}, {'coeff': 46, 'attr': 'imag'}, {'coeff': 47, 'attr': 'imag'}, {'coeff': 48, 'attr': 'imag'}, {'coeff': 49, 'attr': 'imag'}, {'coeff': 50, 'attr': 'imag'}, {'coeff': 51, 'attr': 'imag'}, {'coeff': 52, 'attr': 'imag'}, {'coeff': 53, 'attr': 'imag'}, {'coeff': 54, 'attr': 'imag'}, {'coeff': 55, 'attr': 'imag'}, {'coeff': 56, 'attr': 'imag'}, {'coeff': 57, 'attr': 'imag'}, {'coeff': 58, 'attr': 'imag'}, {'coeff': 59, 'attr': 'imag'}, {'coeff': 60, 'attr': 'imag'}, {'coeff': 61, 'attr': 'imag'}, {'coeff': 62, 'attr': 'imag'}, {'coeff': 63, 'attr': 'imag'}, {'coeff': 64, 'attr': 'imag'}, {'coeff': 65, 'attr': 'imag'}, {'coeff': 66, 'attr': 'imag'}, {'coeff': 67, 'attr': 'imag'}, {'coeff': 68, 'attr': 'imag'}, {'coeff': 69, 'attr': 'imag'}, {'coeff': 70, 'attr': 'imag'}, {'coeff': 71, 'attr': 'imag'}, {'coeff': 72, 'attr': 'imag'}, {'coeff': 73, 'attr': 'imag'}, {'coeff': 74, 'attr': 'imag'}, {'coeff': 75, 'attr': 'imag'}, {'coeff': 76, 'attr': 'imag'}, {'coeff': 77, 'attr': 'imag'}, {'coeff': 78, 'attr': 'imag'}, {'coeff': 79, 'attr': 'imag'}, {'coeff': 80, 'attr': 'imag'}, {'coeff': 81, 'attr': 'imag'}, {'coeff': 82, 'attr': 'imag'}, {'coeff': 83, 'attr': 'imag'}, {'coeff': 84, 'attr': 'imag'}, {'coeff': 85, 'attr': 'imag'}, {'coeff': 86, 'attr': 'imag'}, {'coeff': 87, 'attr': 'imag'}, {'coeff': 88, 'attr': 'imag'}, {'coeff': 89, 'attr': 'imag'}, {'coeff': 90, 'attr': 'imag'}, {'coeff': 91, 'attr': 'imag'}, {'coeff': 92, 'attr': 'imag'}, {'coeff': 93, 'attr': 'imag'}, {'coeff': 94, 'attr': 'imag'}, {'coeff': 95, 'attr': 'imag'}, {'coeff': 96, 'attr': 'imag'}, {'coeff': 97, 'attr': 'imag'}, {'coeff': 98, 'attr': 'imag'}, {'coeff': 99, 'attr': 'imag'}, {'coeff': 0, 'attr': 'abs'}, {'coeff': 1, 'attr': 'abs'}, {'coeff': 2, 'attr': 'abs'}, {'coeff': 3, 'attr': 'abs'}, {'coeff': 4, 'attr': 'abs'}, {'coeff': 5, 'attr': 'abs'}, {'coeff': 6, 'attr': 'abs'}, {'coeff': 7, 'attr': 'abs'}, {'coeff': 8, 'attr': 'abs'}, {'coeff': 9, 'attr': 'abs'}, {'coeff': 10, 'attr': 'abs'}, {'coeff': 11, 'attr': 'abs'}, {'coeff': 12, 'attr': 'abs'}, {'coeff': 13, 'attr': 'abs'}, {'coeff': 14, 'attr': 'abs'}, {'coeff': 15, 'attr': 'abs'}, {'coeff': 16, 'attr': 'abs'}, {'coeff': 17, 'attr': 'abs'}, {'coeff': 18, 'attr': 'abs'}, {'coeff': 19, 'attr': 'abs'}, {'coeff': 20, 'attr': 'abs'}, {'coeff': 21, 'attr': 'abs'}, {'coeff': 22, 'attr': 'abs'}, {'coeff': 23, 'attr': 'abs'}, {'coeff': 24, 'attr': 'abs'}, {'coeff': 25, 'attr': 'abs'}, {'coeff': 26, 'attr': 'abs'}, {'coeff': 27, 'attr': 'abs'}, {'coeff': 28, 'attr': 'abs'}, {'coeff': 29, 'attr': 'abs'}, {'coeff': 30, 'attr': 'abs'}, {'coeff': 31, 'attr': 'abs'}, {'coeff': 32, 'attr': 'abs'}, {'coeff': 33, 'attr': 'abs'}, {'coeff': 34, 'attr': 'abs'}, {'coeff': 35, 'attr': 'abs'}, {'coeff': 36, 'attr': 'abs'}, {'coeff': 37, 'attr': 'abs'}, {'coeff': 38, 'attr': 'abs'}, {'coeff': 39, 'attr': 'abs'}, {'coeff': 40, 'attr': 'abs'}, {'coeff': 41, 'attr': 'abs'}, {'coeff': 42, 'attr': 'abs'}, {'coeff': 43, 'attr': 'abs'}, {'coeff': 44, 'attr': 'abs'}, {'coeff': 45, 'attr': 'abs'}, {'coeff': 46, 'attr': 'abs'}, {'coeff': 47, 'attr': 'abs'}, {'coeff': 48, 'attr': 'abs'}, {'coeff': 49, 'attr': 'abs'}, {'coeff': 50, 'attr': 'abs'}, {'coeff': 51, 'attr': 'abs'}, {'coeff': 52, 'attr': 'abs'}, {'coeff': 53, 'attr': 'abs'}, {'coeff': 54, 'attr': 'abs'}, {'coeff': 55, 'attr': 'abs'}, {'coeff': 56, 'attr': 'abs'}, {'coeff': 57, 'attr': 'abs'}, {'coeff': 58, 'attr': 'abs'}, {'coeff': 59, 'attr': 'abs'}, {'coeff': 60, 'attr': 'abs'}, {'coeff': 61, 'attr': 'abs'}, {'coeff': 62, 'attr': 'abs'}, {'coeff': 63, 'attr': 'abs'}, {'coeff': 64, 'attr': 'abs'}, {'coeff': 65, 'attr': 'abs'}, {'coeff': 66, 'attr': 'abs'}, {'coeff': 67, 'attr': 'abs'}, {'coeff': 68, 'attr': 'abs'}, {'coeff': 69, 'attr': 'abs'}, {'coeff': 70, 'attr': 'abs'}, {'coeff': 71, 'attr': 'abs'}, {'coeff': 72, 'attr': 'abs'}, {'coeff': 73, 'attr': 'abs'}, {'coeff': 74, 'attr': 'abs'}, {'coeff': 75, 'attr': 'abs'}, {'coeff': 76, 'attr': 'abs'}, {'coeff': 77, 'attr': 'abs'}, {'coeff': 78, 'attr': 'abs'}, {'coeff': 79, 'attr': 'abs'}, {'coeff': 80, 'attr': 'abs'}, {'coeff': 81, 'attr': 'abs'}, {'coeff': 82, 'attr': 'abs'}, {'coeff': 83, 'attr': 'abs'}, {'coeff': 84, 'attr': 'abs'}, {'coeff': 85, 'attr': 'abs'}, {'coeff': 86, 'attr': 'abs'}, {'coeff': 87, 'attr': 'abs'}, {'coeff': 88, 'attr': 'abs'}, {'coeff': 89, 'attr': 'abs'}, {'coeff': 90, 'attr': 'abs'}, {'coeff': 91, 'attr': 'abs'}, {'coeff': 92, 'attr': 'abs'}, {'coeff': 93, 'attr': 'abs'}, {'coeff': 94, 'attr': 'abs'}, {'coeff': 95, 'attr': 'abs'}, {'coeff': 96, 'attr': 'abs'}, {'coeff': 97, 'attr': 'abs'}, {'coeff': 98, 'attr': 'abs'}, {'coeff': 99, 'attr': 'abs'}, {'coeff': 0, 'attr': 'angle'}, {'coeff': 1, 'attr': 'angle'}, {'coeff': 2, 'attr': 'angle'}, {'coeff': 3, 'attr': 'angle'}, {'coeff': 4, 'attr': 'angle'}, {'coeff': 5, 'attr': 'angle'}, {'coeff': 6, 'attr': 'angle'}, {'coeff': 7, 'attr': 'angle'}, {'coeff': 8, 'attr': 'angle'}, {'coeff': 9, 'attr': 'angle'}, {'coeff': 10, 'attr': 'angle'}, {'coeff': 11, 'attr': 'angle'}, {'coeff': 12, 'attr': 'angle'}, {'coeff': 13, 'attr': 'angle'}, {'coeff': 14, 'attr': 'angle'}, {'coeff': 15, 'attr': 'angle'}, {'coeff': 16, 'attr': 'angle'}, {'coeff': 17, 'attr': 'angle'}, {'coeff': 18, 'attr': 'angle'}, {'coeff': 19, 'attr': 'angle'}, {'coeff': 20, 'attr': 'angle'}, {'coeff': 21, 'attr': 'angle'}, {'coeff': 22, 'attr': 'angle'}, {'coeff': 23, 'attr': 'angle'}, {'coeff': 24, 'attr': 'angle'}, {'coeff': 25, 'attr': 'angle'}, {'coeff': 26, 'attr': 'angle'}, {'coeff': 27, 'attr': 'angle'}, {'coeff': 28, 'attr': 'angle'}, {'coeff': 29, 'attr': 'angle'}, {'coeff': 30, 'attr': 'angle'}, {'coeff': 31, 'attr': 'angle'}, {'coeff': 32, 'attr': 'angle'}, {'coeff': 33, 'attr': 'angle'}, {'coeff': 34, 'attr': 'angle'}, {'coeff': 35, 'attr': 'angle'}, {'coeff': 36, 'attr': 'angle'}, {'coeff': 37, 'attr': 'angle'}, {'coeff': 38, 'attr': 'angle'}, {'coeff': 39, 'attr': 'angle'}, {'coeff': 40, 'attr': 'angle'}, {'coeff': 41, 'attr': 'angle'}, {'coeff': 42, 'attr': 'angle'}, {'coeff': 43, 'attr': 'angle'}, {'coeff': 44, 'attr': 'angle'}, {'coeff': 45, 'attr': 'angle'}, {'coeff': 46, 'attr': 'angle'}, {'coeff': 47, 'attr': 'angle'}, {'coeff': 48, 'attr': 'angle'}, {'coeff': 49, 'attr': 'angle'}, {'coeff': 50, 'attr': 'angle'}, {'coeff': 51, 'attr': 'angle'}, {'coeff': 52, 'attr': 'angle'}, {'coeff': 53, 'attr': 'angle'}, {'coeff': 54, 'attr': 'angle'}, {'coeff': 55, 'attr': 'angle'}, {'coeff': 56, 'attr': 'angle'}, {'coeff': 57, 'attr': 'angle'}, {'coeff': 58, 'attr': 'angle'}, {'coeff': 59, 'attr': 'angle'}, {'coeff': 60, 'attr': 'angle'}, {'coeff': 61, 'attr': 'angle'}, {'coeff': 62, 'attr': 'angle'}, {'coeff': 63, 'attr': 'angle'}, {'coeff': 64, 'attr': 'angle'}, {'coeff': 65, 'attr': 'angle'}, {'coeff': 66, 'attr': 'angle'}, {'coeff': 67, 'attr': 'angle'}, {'coeff': 68, 'attr': 'angle'}, {'coeff': 69, 'attr': 'angle'}, {'coeff': 70, 'attr': 'angle'}, {'coeff': 71, 'attr': 'angle'}, {'coeff': 72, 'attr': 'angle'}, {'coeff': 73, 'attr': 'angle'}, {'coeff': 74, 'attr': 'angle'}, {'coeff': 75, 'attr': 'angle'}, {'coeff': 76, 'attr': 'angle'}, {'coeff': 77, 'attr': 'angle'}, {'coeff': 78, 'attr': 'angle'}, {'coeff': 79, 'attr': 'angle'}, {'coeff': 80, 'attr': 'angle'}, {'coeff': 81, 'attr': 'angle'}, {'coeff': 82, 'attr': 'angle'}, {'coeff': 83, 'attr': 'angle'}, {'coeff': 84, 'attr': 'angle'}, {'coeff': 85, 'attr': 'angle'}, {'coeff': 86, 'attr': 'angle'}, {'coeff': 87, 'attr': 'angle'}, {'coeff': 88, 'attr': 'angle'}, {'coeff': 89, 'attr': 'angle'}, {'coeff': 90, 'attr': 'angle'}, {'coeff': 91, 'attr': 'angle'}, {'coeff': 92, 'attr': 'angle'}, {'coeff': 93, 'attr': 'angle'}, {'coeff': 94, 'attr': 'angle'}, {'coeff': 95, 'attr': 'angle'}, {'coeff': 96, 'attr': 'angle'}, {'coeff': 97, 'attr': 'angle'}, {'coeff': 98, 'attr': 'angle'}, {'coeff': 99, 'attr': 'angle'}],
            'fft_aggregated': [{'aggtype': 'centroid'}, {'aggtype': 'variance'}, {'aggtype': 'skew'}, {'aggtype': 'kurtosis'}],
            'quantile': [{'q': 0.1}, {'q': 0.2}, {'q': 0.3}, {'q': 0.4}, {'q': 0.6}, {'q': 0.7}, {'q': 0.8}, {'q': 0.9}],  # Change of quantile (QT)
            # 'approximate_entropy': [{'m': 2, 'r': 0.1}, {'m': 2, 'r': 0.3}, {'m': 2, 'r': 0.5}, {'m': 2, 'r': 0.7}, {'m': 2, 'r': 0.9}],
            "skewness": None,  # Sample skewness (SKE)
            "kurtosis": None,  # Sample kurtosis (KUR)
            "count_above_mean": None,  # Length above Mean
            "count_below_mean": None,  # Length below Mean
            "longest_strike_above_mean": None,
            "longest_strike_below_mean": None,
            'agg_linear_trend': [{'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'min'}, {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'rvalue', 'chunk_len': 5, 'f_agg': 'var'}, {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'max'}, {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'min'}, {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'rvalue', 'chunk_len': 10, 'f_agg': 'var'}, {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'max'}, {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'min'}, {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'mean'}, {'attr': 'rvalue', 'chunk_len': 50, 'f_agg': 'var'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'min'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'intercept', 'chunk_len': 5, 'f_agg': 'var'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'min'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'intercept', 'chunk_len': 10, 'f_agg': 'var'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'max'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'min'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'mean'}, {'attr': 'intercept', 'chunk_len': 50, 'f_agg': 'var'}, {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'min'}, {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'slope', 'chunk_len': 5, 'f_agg': 'var'}, {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'max'}, {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'min'}, {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'slope', 'chunk_len': 10, 'f_agg': 'var'}, {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'max'}, {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'min'}, {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'mean'}, {'attr': 'slope', 'chunk_len': 50, 'f_agg': 'var'}, {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'max'}, {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'min'}, {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'mean'}, {'attr': 'stderr', 'chunk_len': 5, 'f_agg': 'var'}, {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'max'}, {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'min'}, {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'mean'}, {'attr': 'stderr', 'chunk_len': 10, 'f_agg': 'var'}, {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'max'}, {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'min'}, {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'mean'}, {'attr': 'stderr', 'chunk_len': 50, 'f_agg': 'var'}],
            'change_quantiles': [{'ql': 0.0, 'qh': 0.2, 'isabs': False, 'f_agg': 'mean'}, {'ql': 0.0, 'qh': 0.2, 'isabs': False, 'f_agg': 'var'}, {'ql': 0.0, 'qh': 0.2, 'isabs': True, 'f_agg': 'mean'}, {'ql': 0.0, 'qh': 0.2, 'isabs': True, 'f_agg': 'var'}, {'ql': 0.0, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'}, {'ql': 0.0, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'}, {'ql': 0.0, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'}, {'ql': 0.0, 'qh': 0.4, 'isabs': True, 'f_agg': 'var'}, {'ql': 0.0, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'}, {'ql': 0.0, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'}, {'ql': 0.0, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'}, {'ql': 0.0, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'}, {'ql': 0.0, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'}, {'ql': 0.0, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'}, {'ql': 0.0, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'}, {'ql': 0.0, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'}, {'ql': 0.0, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'}, {'ql': 0.0, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'}, {'ql': 0.0, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'}, {'ql': 0.0, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'}, {'ql': 0.2, 'qh': 0.4, 'isabs': False, 'f_agg': 'mean'}, {'ql': 0.2, 'qh': 0.4, 'isabs': False, 'f_agg': 'var'}, {'ql': 0.2, 'qh': 0.4, 'isabs': True, 'f_agg': 'mean'}, {'ql': 0.2, 'qh': 0.4, 'isabs': True, 'f_agg': 'var'}, {'ql': 0.2, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'}, {'ql': 0.2, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'}, {'ql': 0.2, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'}, {'ql': 0.2, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'}, {'ql': 0.2, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'}, {'ql': 0.2, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'}, {'ql': 0.2, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'}, {'ql': 0.2, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'}, {'ql': 0.2, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'}, {'ql': 0.2, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'}, {'ql': 0.2, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'}, {'ql': 0.2, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'}, {'ql': 0.4, 'qh': 0.6, 'isabs': False, 'f_agg': 'mean'}, {'ql': 0.4, 'qh': 0.6, 'isabs': False, 'f_agg': 'var'}, {'ql': 0.4, 'qh': 0.6, 'isabs': True, 'f_agg': 'mean'}, {'ql': 0.4, 'qh': 0.6, 'isabs': True, 'f_agg': 'var'}, {'ql': 0.4, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'}, {'ql': 0.4, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'}, {'ql': 0.4, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'}, {'ql': 0.4, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'}, {'ql': 0.4, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'}, {'ql': 0.4, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'}, {'ql': 0.4, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'}, {'ql': 0.4, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'}, {'ql': 0.6, 'qh': 0.8, 'isabs': False, 'f_agg': 'mean'}, {'ql': 0.6, 'qh': 0.8, 'isabs': False, 'f_agg': 'var'}, {'ql': 0.6, 'qh': 0.8, 'isabs': True, 'f_agg': 'mean'}, {'ql': 0.6, 'qh': 0.8, 'isabs': True, 'f_agg': 'var'}, {'ql': 0.6, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'}, {'ql': 0.6, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'}, {'ql': 0.6, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'}, {'ql': 0.6, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'}, {'ql': 0.8, 'qh': 1.0, 'isabs': False, 'f_agg': 'mean'}, {'ql': 0.8, 'qh': 1.0, 'isabs': False, 'f_agg': 'var'}, {'ql': 0.8, 'qh': 1.0, 'isabs': True, 'f_agg': 'mean'}, {'ql': 0.8, 'qh': 1.0, 'isabs': True, 'f_agg': 'var'}],
        }


        #Extract features from training and testing data
        print("Extracting features from training data...")
        train_features = extract_features(train_df, column_id="id", column_sort="time",default_fc_parameters=default_fc_parameters1)
        print(train_features)
        print("Extracting features from testing data...")
        test_features = extract_features(test_df, column_id="id", column_sort="time",default_fc_parameters=default_fc_parameters1)
        print(test_features)


        # Standardize the data
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

        # print("Extracting features from training data...")
        # train_features = extract_features(train_df, column_id="id", column_sort="time")
        # print(train_features)
        # print("Extracting features from testing data...")
        # test_features = extract_features(test_df, column_id="id", column_sort="time")
        # print(test_features)

        # #possible to add feature selection for training set, and choose the same features for test.
        # train_features = select_features(train_features,  train_labels)
        # # Get the selected feature columns from the training set
        # selected_columns = train_features.columns
        # print(train_features)
        # # Filter the test features to keep only the selected columns
        # test_features = test_features[selected_columns]
        # print(test_features)

        #add xgboost
        # Train models
        models = {
            "Decision Tree (balanced)": DecisionTreeClassifier(random_state=42,class_weight="balanced"),
            "Logistic Regression (balanced)": LogisticRegression(class_weight="balanced"),
            "SVM (balanced) ": SVC(kernel='linear', class_weight='balanced', random_state=42),
        }
        results = []

        for model_name, model in models.items():
            print(f"\nTraining and evaluating model: {model_name}")
            model.fit(train_features, train_labels)
            y_pred = model.predict(test_features)

            # Calculate metrics
            accuracy = accuracy_score(test_labels, y_pred)
            precision = precision_score(test_labels, y_pred, pos_label=1)
            recall = recall_score(test_labels, y_pred, pos_label=1)
            f1 = f1_score(test_labels, y_pred, pos_label=1)
            report = classification_report(test_labels, y_pred, target_names=["Class 0", "Class 1"])
            print(report)


            # Ensure you have probabilities for AUC calculation
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(test_features)[:, 1]  # Probability for the positive class
                auc = roc_auc_score(test_labels, y_pred_proba)
            elif hasattr(model, "decision_function"):  # For models like SVM
                y_pred_proba = model.decision_function(test_features)
                auc = roc_auc_score(test_labels, y_pred_proba)
            else:
                auc = None  # AUC cannot be calculated
            
            results.append({
                "Interpolation Method": interpolation_type,
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "AUC": auc
            })

        # Append the results for this interpolation method
        results_all.extend(results)

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results_all)
    print("\nEvaluation Results:")
    print(results_df)
    return results_df

# Example usage
if __name__ == "__main__":
    zip_file_path = r'/sise/home/mayaroz/signals.zip'
    results_df = evaluate_models(zip_file_path, class_dict)
    results_df.to_csv("classification_results.csv")

