import numpy as np
import pandas as pd
import zipfile
import os
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from tabulate import tabulate



def analyze_missing_values(zip_path):
    results = []
    temp_dir = "temp_extracted"  # Temporary directory to extract files
    missing_rates = []
    missing_sequence_lengths = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(temp_dir)
        nested_path = os.path.join(temp_dir, "signals", "signals")

        # Iterate through all CSV files in the nested directory
        for root, _, files in os.walk(nested_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    try:
                        data = pd.read_csv(file_path)
                        fhr = data['FHR'].values      
                        fhr = np.where(fhr == 0, np.nan, fhr)
                        # Calculate missing rate for the file
                        missing_rate = np.mean(pd.isna(fhr)) * 100
                        missing_rates.append(missing_rate)    
                        # Find consecutive missing sequences
                        missing_indices = np.where(pd.isna(fhr))[0]
                        if len(missing_indices) > 0:
                            diff = np.diff(missing_indices)
                            missing_sequences = np.split(missing_indices, np.where(diff != 1)[0] + 1)
                            missing_sequence_lengths.extend([len(seq) for seq in missing_sequences])
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        avg_missing_rate =  np.mean(missing_rates)
        median_missing_rate = np.median(missing_rates)
        avg_missing_sequence_lengths = np.mean(missing_sequence_lengths)
        median_missing_sequence_lengths = np.median(missing_sequence_lengths)
        return missing_rates, missing_sequence_lengths ,  avg_missing_rate, avg_missing_sequence_lengths , median_missing_rate , median_missing_sequence_lengths



def plot_missing_data_statistics(missing_rates, missing_sequence_lengths,output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    missing_rates_mean = np.mean(missing_rates)
    missing_rates_median = np.median(missing_rates)
    missing_rates_std = np.std(missing_rates)
    missing_rates_min = np.min(missing_rates)
    missing_rates_max = np.max(missing_rates)
    missing_rates_25th = np.percentile(missing_rates, 25)
    missing_rates_75th = np.percentile(missing_rates, 75)

    missing_sequences_mean = np.mean(missing_sequence_lengths)
    missing_sequences_median = np.median(missing_sequence_lengths)
    missing_sequences_std = np.std(missing_sequence_lengths)
    missing_sequences_min = np.min(missing_sequence_lengths)
    missing_sequences_max = np.max(missing_sequence_lengths)
    missing_sequences_25th = np.percentile(missing_sequence_lengths, 25)
    missing_sequences_75th = np.percentile(missing_sequence_lengths, 75)

    stats_dict = {
        'Metric': ['Mean', 'Median', 'Standard Deviation', 'Min', 'Max', '25th Percentile', '75th Percentile'],
        'Missing Rates (%)': [
            missing_rates_mean, 
            missing_rates_median, 
            missing_rates_std, 
            missing_rates_min, 
            missing_rates_max, 
            missing_rates_25th, 
            missing_rates_75th
        ],
        'Missing Sequence Lengths': [
            missing_sequences_mean, 
            missing_sequences_median, 
            missing_sequences_std, 
            missing_sequences_min, 
            missing_sequences_max, 
            missing_sequences_25th, 
            missing_sequences_75th
        ]
    }
    
    stats_df = pd.DataFrame(stats_dict)
    stats_df = stats_df.round(2)
    stats_df = stats_df.set_index('Metric').T 


    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.axis('off')  
    table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns, rowLabels=stats_df.index, loc='center', cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)  

    plt.savefig(f'{output_dir}/missing_data_statistics_table.jpg', format='jpg', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nMissing Data Statistics:\n")
    print(tabulate(stats_df, headers='keys', tablefmt='fancy_grid', showindex=True, numalign='decimal'))
    

    # Plot histogram for missing rates across files
    plt.figure(figsize=(10, 6))
    plt.hist(missing_rates, bins=20, color='skyblue')
    plt.title('Distribution of Missing Data Proportions Across Files')
    plt.xlabel('Missing Data (%)')
    plt.ylabel('Frequency')
    plt.savefig(f'{output_dir}/missing_data_distribution.jpg', format='jpg', dpi=300)  # Save as JPG
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(missing_sequence_lengths, bins=20, color="skyblue")
    plt.title('Distribution of Missing Data Sequence Lengths')
    plt.xlabel('Missing Data Sequence Lengths')
    plt.ylabel('Frequency')
    plt.xlim(0,1000)
    plt.savefig(f'{output_dir}/missing_sequence_lengths_distribution.jpg', format='jpg', dpi=300)
    plt.close()



if __name__ == "__main__":
    zip_file_path = r'/sise/home/mayaroz/signals.zip'
    missing_rates, missing_sequence_lengths ,  avg_missing_rate, avg_missing_sequence_lengths , median_missing_rate , median_missing_sequence_lengths = analyze_missing_values(zip_file_path)
    plot_missing_data_statistics(missing_rates,missing_sequence_lengths)
