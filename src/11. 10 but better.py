import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to convert Carrington map number to month-year string
def get_month_year_from_map_number(map_number):
    base_date = datetime(1853, 11, 9)  # Base date for Carrington rotation 1
    rotation_period_days = 27.2753  # Average Carrington rotation period in days
    days_since_base = (int(map_number) - 1) * rotation_period_days
    map_date = base_date + timedelta(days=days_since_base)
    return map_date.strftime("%B %Y")

# Function to compute the average absolute solar flux from CSV
def compute_avg_flux(csv_file):
    # Read the magnetic field data from CSV
    b = np.loadtxt(csv_file, delimiter=",")
    
    # Handle NaN and infinite values if any
    b = np.nan_to_num(b, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    
    # Compute the average absolute magnetic flux
    mean_flux = np.mean(np.abs(b))
    
    # Extract Carrington Rotation number from the filename
    cr_number = os.path.basename(csv_file).split("_CR")[1].split(".csv")[0]
    
    return int(cr_number), mean_flux

# Path where CR CSV files are stored
csv_folder = "magnetic fields"  # <-- Change this to the actual path
csv_files = glob.glob(os.path.join(csv_folder, "magnetic_field_CR*.csv"))

# Storage lists
cr_numbers = []
avg_flux_values = []
time_labels = []

# Process each CR map CSV file
for csv_file in sorted(csv_files):
    cr, avg_flux = compute_avg_flux(csv_file)
    cr_numbers.append(cr)
    avg_flux_values.append(avg_flux)
    time_labels.append(get_month_year_from_map_number(cr))  # Convert CR number to Month-Year
    # print(f"CR {cr} ({time_labels[-1]}): Avg Magnetic Flux = {avg_flux:.2e}")

# Extract only one September per year for x-axis labels
september_labels = []
cr_numbers_filtered = []
seen_years = set()  # To keep track of the years we've already added a September for

for i, label in enumerate(time_labels):
    if "September" in label:  # Filter for September
        year = label.split()[-1]  # Extract the year from the label
        if year not in seen_years:
            september_labels.append(label)
            cr_numbers_filtered.append(cr_numbers[i])
            seen_years.add(year)  # Mark this year as having a September already

# Plot the computed total solar flux over time
plt.figure(figsize=(12, 6))
plt.plot(cr_numbers, avg_flux_values, linestyle="-", color="blue", label=r"$|B_{avg}|$")

plt.xlabel("Time (Carrington Rotation Number & Month-Year)")
plt.ylabel("Average Absolute Magnetic Flux (Gauss)")
plt.title("Average Solar Magnetic Flux Variation Over Time")

# Show only one September per year on the x-axis
plt.xticks(ticks=cr_numbers_filtered, labels=september_labels, rotation=45)
plt.legend()
plt.grid()
plt.show()