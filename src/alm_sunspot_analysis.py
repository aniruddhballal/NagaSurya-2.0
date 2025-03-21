import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from datetime import datetime
import pywt
from scipy.stats import pearsonr

def get_sunspot_data():
    sunspot_file = 'sunspot_numbers.csv'  # Ensure this is the correct file path
    df = pd.read_csv(sunspot_file, sep=';', header=None, 
                     names=["Year", "Month", "Decimal_Year", "SunspotNumber", "StdDev", "Observations", "Definitive"])

    # Strip any whitespace in column names (just in case)
    df.columns = df.columns.str.strip()

    # Drop rows with missing values in relevant columns
    df = df[["Year", "Month", "SunspotNumber"]].dropna()

    # Convert Year and Month to datetime
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(day=1), errors='coerce')

    # Drop any rows where the date could not be parsed
    df = df.dropna(subset=['Date'])

    return df[['Date', 'SunspotNumber']]


def get_month_year_from_map_number(map_number):
    """Converts Carrington Rotation number to approximate month-year."""
    start_cr = 2096  # First CR in your dataset
    start_date = datetime(2010, 5, 1)  # Approximate start date of CR 2096
    cr_length_days = 27.2753  # Average Carrington rotation length
    delta_days = (int(map_number) - start_cr) * cr_length_days
    return (start_date + pd.to_timedelta(delta_days, unit='D')).strftime('%B %Y')

def process_all_csv_files():
    folder_name = 'alm_values'
    percentiles = [95, 99, 99.5]  # Different percentile thresholds

    sunspot_data = get_sunspot_data()
    avg_values_dict = {perc: [] for perc in percentiles}
    months_years = []

    for csv_file in os.listdir(folder_name):
        if csv_file.endswith('.csv'):
            map_number = csv_file.split('_')[1].split('.')[0]
            csv_filename = os.path.join(folder_name, csv_file)

            df = pd.read_csv(csv_filename)
            alm_values = df['alm'].apply(lambda x: complex(x.replace('(', '').replace(')', ''))).values
            alm_magnitudes = np.abs(alm_values)

            for perc in percentiles:
                threshold = np.percentile(alm_magnitudes, perc)
                values_above_threshold = alm_magnitudes[alm_magnitudes > threshold]
                avg_values_dict[perc].append(np.mean(values_above_threshold))

            month_year = get_month_year_from_map_number(map_number)
            months_years.append(month_year)

    months_years_dt = [datetime.strptime(m, "%B %Y") for m in months_years]
    
    # Merge with sunspot data
    results_df = pd.DataFrame({'Date': months_years_dt})
    for perc in percentiles:
        results_df[f'alm_avg_{perc}'] = gaussian_filter(avg_values_dict[perc], sigma=2)
    
    results_df = results_df.merge(sunspot_data, on='Date', how='inner')

    # Compute Correlations
    correlations = {}
    for perc in percentiles:
        correlation, _ = pearsonr(results_df[f'alm_avg_{perc}'], results_df['SunspotNumber'])
        correlations[perc] = correlation
    print("Correlations with Sunspot Number:", correlations)

    # Create and save the combined plot
    plot_combined_alm_vs_sunspots(results_df, percentiles)

    # Perform Wavelet Transform
    perform_wavelet_analysis(results_df, percentiles)

def plot_combined_alm_vs_sunspots(results_df, percentiles):
    """Plots all three avg alm values and sunspot numbers with two y-axes"""
    fig, ax1 = plt.subplots(figsize=(12, 6))

    colors = ['b', 'g', 'c']  # Colors for 95%, 99%, 99.5% alm values

    # Plot alm magnitudes for all percentiles
    for i, perc in enumerate(percentiles):
        ax1.plot(results_df['Date'], results_df[f'alm_avg_{perc}'], marker='o', linestyle='-', color=colors[i], label=f'alm {perc}%')

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Average alm Magnitude', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot sunspot numbers on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(results_df['Date'], results_df['SunspotNumber'], marker='s', linestyle='--', color='r', label='Sunspot Number')
    ax2.set_ylabel('Sunspot Number', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Grid and legend
    plt.title('alm Magnitude vs. Sunspot Cycle')
    ax1.grid(True)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Save plot
    plot_filename = 'plots/alm_vs_sunspots_combined.png'
    plt.savefig(plot_filename)
    plt.close()
    print(f"Plot saved as {plot_filename}")

def perform_wavelet_analysis(results_df, percentiles):
    """Performs wavelet transform on alm magnitudes"""
    scales = np.arange(1, 128)  # Define scales for wavelet transform

    for perc in percentiles:
        alm_series = results_df[f'alm_avg_{perc}'].values

        # Continuous Wavelet Transform
        coefficients, frequencies = pywt.cwt(alm_series, scales, 'cmor')

        # Plot Wavelet Transform
        plt.figure(figsize=(12, 6))
        plt.imshow(np.abs(coefficients), aspect='auto', extent=[results_df['Date'].min(), results_df['Date'].max(), scales.min(), scales.max()], cmap='jet')
        plt.colorbar(label='Wavelet Power')
        plt.ylabel('Periodicity (Scales)')
        plt.xlabel('Year')
        plt.title(f'Wavelet Transform of alm Magnitude ({perc}%)')

        # Save wavelet plot
        wavelet_filename = f'plots/wavelet_alm_{perc}.png'
        plt.savefig(wavelet_filename)
        plt.close()
        print(f"Wavelet plot saved as {wavelet_filename}")

process_all_csv_files()