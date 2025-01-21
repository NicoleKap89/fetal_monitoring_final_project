import math
import os
import zipfile
import random
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, CubicHermiteSpline
from sklearn.metrics import mean_squared_error
import copy
import matplotlib.pyplot as plt
from tabulate import tabulate

def process_signal(file_path, masking_rate_pct, random_selection):
    # Read the data
    data = pd.read_csv(file_path)  # df
    timestamps = data['seconds'].values  # ndarray
    fhr = data['FHR'].values  # ndarray

    # Replace zeros with NaN
    fhr = np.where(fhr == 0, np.nan, fhr)  # ndarray
 
    null_rate = np.mean(np.isnan(fhr)) * 100  # Percentage of NaN values in fhr
    
    # if null_rate > 44: #95 quantile
    #     print(file_path)
    #     return None
    
    original_fhr = copy.copy(fhr) #without masking nulls
    # Identify valid indices
    valid_indices = np.where(~np.isnan(fhr))[0]  # Indices where FHR is not NaN
    num_masked_values = int(len(valid_indices) * masking_rate_pct)

    # Apply masking
    if random_selection==True:
        # Randomly select indices to mask
        masked_indices = random.sample(list(valid_indices), num_masked_values)
    else:
        # Consecutive masking in random segments
        segment_size = 8
        number_of_segments = num_masked_values // segment_size
        remainder = num_masked_values % segment_size
        segment_sizes = [segment_size] * number_of_segments + ([remainder] if remainder > 0 else [])
        masked_indices = []
        used_starts = set()

        for size in segment_sizes:
            stopping= 0
            while True:
                start_idx = random.choice(valid_indices[:-size])
                segment = range(start_idx, start_idx + size)
                # Ensure the segment doesn't overlap with existing segments
                if not any(idx in masked_indices for idx in segment):
                    masked_indices.extend(segment)
                    used_starts.add(start_idx)
                    break
                stopping+=1
                if stopping > 1000:
                    raise ValueError("Maya's & Nicloe's Error - Cannot support this masking rate for conscutive segments")


    # Create a boolean mask for artificially applied nulls
    artificial_mask = np.zeros_like(fhr, dtype=bool)
    artificial_mask[masked_indices] = True

    # Store original values for evaluation
    original_values = fhr[masked_indices]
    # Apply masking
    fhr[masked_indices] = np.nan

    linear_fhr = copy.copy(fhr)
    cubic_fhr = copy.copy(fhr)
    linear_fhr = pd.DataFrame({'FHR': linear_fhr})
    cubic_fhr = pd.DataFrame({'FHR': cubic_fhr})
    # Linear interpolation
    linear_interpolated = linear_fhr['FHR'].interpolate(method='linear', limit_direction='both')

    # Cubic spline interpolation
    cubic_interpolated = cubic_fhr['FHR'].interpolate(method='cubic', limit_direction='both', fill_value='extrapolate')
    
    hermite_fhr = pd.DataFrame({'FHR': fhr, 'seconds': timestamps})
    known_indices = ~hermite_fhr['FHR'].isna()
    known_timestamps = hermite_fhr.loc[known_indices, 'seconds'].values
    known_values = hermite_fhr.loc[known_indices, 'FHR'].values
    missing_indices = hermite_fhr['FHR'].isna()
    missing_timestamps = hermite_fhr.loc[missing_indices, 'seconds'].values
    known_derivatives = calculate_derivatives(known_timestamps, known_values)
    hermite_spline = CubicHermiteSpline(known_timestamps, known_values, known_derivatives,extrapolate='periodic')
    hermite_interpolated = hermite_spline(missing_timestamps)
    hermite_fhr.loc[missing_indices, 'FHR'] = hermite_interpolated
    hermite_interpolated = hermite_fhr['FHR']

    # XOR condition: Masked and originally not null
    valid_eval_indices = np.logical_and(artificial_mask, ~np.isnan(original_fhr))
    # Evaluate imputation performance for valid_eval_indices
    linear_mse = mean_squared_error(
        original_fhr[valid_eval_indices],
        linear_interpolated[valid_eval_indices]
    )
    cubic_mse = mean_squared_error(
        original_fhr[valid_eval_indices],
        cubic_interpolated[valid_eval_indices]
    )
    hermite_mse = mean_squared_error(
        original_fhr[valid_eval_indices],
        hermite_interpolated[valid_eval_indices]
    )

    return {
        "file": file_path,
        "masking rate" : masking_rate_pct,
        "selction method" : random_selection,
        "linear_mse": linear_mse,
        "cubic_mse": cubic_mse,
        "hermite_mse": hermite_mse,
    }


def process_zip(zip_path, masking_rate_pct, random_selection):
    results = []
    temp_dir = "temp_extracted"  # Temporary directory to extract files
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
                        # Process the signal
                        result = process_signal(file_path, masking_rate_pct, random_selection)
                        if result != None:
                            results.append(result)
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

    # Clean up temporary directory
    for root, _, files in os.walk(temp_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        os.rmdir(root)

    return results


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



# Define the combinations of masking rates and selection meSthods
masking_rates = [0.01,0.025,0.05,0.075,0.1,0.125,0.15]
selection_methods = [True, False]


zip_path = r'/sise/home/mayaroz/signals.zip'

# Initialize a list to store summary results
summary_results = []

# Loop through each combination of masking rate and selection method
for masking_rate_pct in masking_rates:
    for random_selection in selection_methods:
        # Process the ZIP file for the current combination
        results = process_zip(zip_path, masking_rate_pct, random_selection)

        # Initialize accumulators for MSE and file count
        all_files = len(results)
        sum_linear_mse = sum(result['linear_mse'] for result in results)
        sum_cubic_mse = sum(result['cubic_mse'] for result in results)
        sum_hermite_mse = sum(result['hermite_mse'] for result in results)

        # Calculate average MSE and RMSE for each interpolation method
        avg_linear_mse = sum_linear_mse / all_files
        avg_cubic_mse = sum_cubic_mse / all_files
        avg_hermite_mse = sum_hermite_mse / all_files

        avg_linear_rmse = math.sqrt(avg_linear_mse)
        avg_cubic_rmse = math.sqrt(avg_cubic_mse)
        avg_hermite_rmse = math.sqrt(avg_hermite_mse)

        # Append the results to the summary list
        summary_results.append({
            "Interpolation Method": "Linear",
            "Masking Rate": masking_rate_pct,
            "Selection Method": "Random" if random_selection else "Sequential",
            "Avg MSE": avg_linear_mse,
            "Avg RMSE": avg_linear_rmse,
        })
        summary_results.append({
            "Interpolation Method": "Cubic",
            "Masking Rate": masking_rate_pct,
            "Selection Method": "Random" if random_selection else "Sequential",
            "Avg MSE": avg_cubic_mse,
            "Avg RMSE": avg_cubic_rmse,
        })
        summary_results.append({
            "Interpolation Method": "Hermite",
            "Masking Rate": masking_rate_pct,
            "Selection Method": "Random" if random_selection else "Sequential",
            "Avg MSE": avg_hermite_mse,
            "Avg RMSE": avg_hermite_rmse,
        })

# Create a structured DataFrame for better representation
structured_data = []

# Loop through the summary results and restructure
for result in summary_results:
    masking_rate = result["Masking Rate"]
    selection_method = result["Selection Method"]
    method = result["Interpolation Method"]
    mse = result["Avg MSE"]
    rmse = result["Avg RMSE"]

    # Append a row for each interpolation method
    structured_data.append({
        "Masking Rate": masking_rate,
        "Selection Method": selection_method,
        "Method": method,
        "MSE": mse,
        "RMSE": rmse
    })

# Convert to a DataFrame
structured_df = pd.DataFrame(structured_data)

# Pivot the DataFrame to the desired structure
final_df = structured_df.pivot_table(
    index=["Masking Rate", "Selection Method"],
    columns="Method",
    values=["MSE", "RMSE"],
    aggfunc="first"
)

# Flatten the multi-index columns for better readability
final_df.columns = [' '.join(col).strip() for col in final_df.columns.values]

# Reset the index for a cleaner look
final_df.reset_index(inplace=True)

final_df = final_df.round(2)

# Display the final structured DataFrame
print(final_df)



# Save the table as a JPG image using matplotlib
def save_table_as_image(df, output_dir, file_name="table_image.jpg"):
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
    ax = plt.gca()
    ax.axis('off')  # Hide the axis
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)  # Adjust table scaling for better readability

    # Save the table as a JPG file
    plt.savefig(f'{output_dir}/{file_name}', format='jpg', dpi=300, bbox_inches='tight')
    plt.close()

# Print the table to the console using tabulate
def print_table(df):
    print("\nFormatted Table:\n")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=True, numalign='decimal'))

# Example usage
output_dir = "plots"  # Change to the directory where you want to save the image
file_name = "interpolation_results.jpg"
final_df.to_csv("output_final_df.csv")
# Save and print the table
save_table_as_image(final_df, output_dir, file_name)
print_table(final_df)




