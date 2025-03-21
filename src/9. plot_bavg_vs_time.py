import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from datetime import datetime, timedelta
# from scipy.ndimage import gaussian_filter
from rich.progress import Progress

# Function to convert Carrington map number to month-year string
def get_month_year_from_map_number(map_number):
    base_date = datetime(1853, 11, 9)  # Base date for Carrington rotation 1
    rotation_period_days = 27.2753  # Average Carrington rotation period in days
    days_since_base = (int(map_number) - 1) * rotation_period_days
    map_date = base_date + timedelta(days=days_since_base)
    return map_date.strftime("%B %Y")

# Directory containing FITS files
fits_dir = "fits files"
plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

# Carrington Rotation range
start_cr = 2096
end_cr = 2293

b_avg_values = []
# b_n_values = []
# b_s_values = []
dates = []
cr_numbers = []

with Progress() as progress:
    task = progress.add_task("Processing FITS files", total=end_cr - start_cr + 1)
    
    for carrington_map_number in range(start_cr, end_cr + 1):
        fits_file = os.path.join(fits_dir, f'hmi.Synoptic_Mr_small.{carrington_map_number}.fits')
        
        try:
            with fits.open(fits_file) as hdul:
                b = hdul[0].data

            if b is None:
                raise ValueError(f"Data missing in file: {fits_file}")

            b = np.nan_to_num(b, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
            # b = gaussian_filter(b, sigma=2)
            b_avg = np.mean(np.abs(b))

            # Define the output CSV file path
            csv_filename = os.path.join(plots_dir, f"magnetic_field_CR{carrington_map_number}.csv")

            # Save the entire 2D array 'b' to a CSV file
            np.savetxt(csv_filename, b, delimiter=",", fmt="%.6f")

            print(f"Magnetic field data saved at: {csv_filename}")

            
            '''
            # Identify North and South magnetic flux based on the sign of 'b'
            north_flux = b[b > 0]   # North magnetic field has positive flux
            south_flux = b[b < 0]   # South magnetic field has negative flux

            # Summing up the north and south magnetic flux magnitudes
            north_flux_sum = np.sum(np.abs(north_flux))  # Summation of the magnitude of north flux
            south_flux_sum = np.sum(np.abs(south_flux))  # Summation of the magnitude of south flux
            '''

            b_avg_values.append(b_avg)
            '''b_n_values.append(north_flux_sum)
            b_s_values.append(south_flux_sum)'''
            dates.append(get_month_year_from_map_number(carrington_map_number))
            cr_numbers.append(carrington_map_number)
        
        except (FileNotFoundError, ValueError) as e:
            continue
            #print(f"Error processing {fits_file}: {e}")
        
        progress.update(task, advance=1)

# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot b_avg on the main y-axis
ax1.plot(cr_numbers, b_avg_values, marker='o', linestyle='-', color='b', label=r'$B_{avg}$')
ax1.set_xlabel("Carrington Rotation Number")
ax1.set_ylabel("Average Magnetic Flux (Gauss)", color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title("Solar Magnetic Flux Variation Over Time")
'''
# Create a second y-axis for North and South magnetic flux
ax2 = ax1.twinx()
ax2.plot(cr_numbers, b_n_values, marker='o', linestyle='-', color='g', label=r'$B_{north}$ (North Flux)')
ax2.plot(cr_numbers, b_s_values, marker='o', linestyle='-', color='r', label=r'$B_{south}$ (South Flux)')
ax2.set_ylabel("Magnetic Flux Magnitude (Gauss)", color='k')
ax2.tick_params(axis='y', labelcolor='k')
'''
# Add legends for both axes
ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')

# Add grid
ax1.grid()

# Show the plot
plt.xticks(ticks=cr_numbers[::20], labels=dates[::20], rotation=45, fontsize=8)
plt.tight_layout()

# Save the plot
plot_path = os.path.join(plots_dir, "solar_magnetic_flux_variation_without_north_south.png")
plt.savefig(plot_path, bbox_inches='tight', dpi=300)
plt.show()