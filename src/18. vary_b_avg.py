import numpy as np
from scipy.special import sph_harm
import csv
import os
from astropy.io import fits
from scipy.integrate import simpson
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def read_alm_from_csv(csv_filename):
    alm = {}
    if os.path.exists(csv_filename):
        with open(csv_filename, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                l = int(row['l'])
                m = int(row['m'])
                alm_real = float(row['alm_real'])
                alm_imag = float(row['alm_imag'])
                alm[(l, m)] = complex(alm_real, alm_imag)  # Reconstructing complex number
    return alm

def write_alm_to_csv(csv_filename, alm):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['l', 'm', 'alm_real', 'alm_imag'])  # Updated headers
        for (l, m), value in alm.items():
            writer.writerow([l, m, value.real, value.imag])  # Storing real and imag parts separately

def calcalm(b_func, lmax, num_points_theta, num_points_phi, carrington_map_number, check):
    alm = {}
    values_folder = ""
    if check == 1:
        values_folder = "alm_values_split"
    else:
        values_folder = "predicted_alm_values"
    '''if not os.path.exists(values_folder):
        os.makedirs(values_folder)'''
    if check == 1:
        values_filename = os.path.join(values_folder, f'values_{carrington_map_number}.csv')
    else:
        values_filename = os.path.join(values_folder, f'predicted_values_{carrington_map_number}.csv')
    alm.update(read_alm_from_csv(values_filename))

    def integrand(theta, phi, l, m):
        ylm = sph_harm(m, l, phi, theta)
        return b_func(theta, phi) * np.conj(ylm) * np.sin(theta)

    theta = np.linspace(0, np.pi, num_points_theta)
    phi = np.linspace(0, 2 * np.pi, num_points_phi)
    dtheta = np.pi / (num_points_theta - 1)
    dphi = 2 * np.pi / (num_points_phi - 1)

    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if (l, m) not in alm:
                theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
                integrand_values = integrand(theta_grid, phi_grid, l, m)
                real_result = simpson(simpson(integrand_values.real, dx=dphi, axis=1), dx=dtheta, axis=0)
                imag_result = simpson(simpson(integrand_values.imag, dx=dphi, axis=1), dx=dtheta, axis=0)
                alm[(l, m)] = real_result + 1j * imag_result

                # Write the current state of alm to CSV
                write_alm_to_csv(values_filename, alm)

    return alm

def save_progress(brecons, save_file, metadata_file, last_l, last_m):
    """Save brecons array and reconstruction progress metadata."""
    np.save(save_file, brecons)  # Save brecons values
    with open(metadata_file, 'w') as f:
        json.dump({'last_l': last_l, 'last_m': last_m}, f)  # Save last computed l, m

def load_progress(save_file, metadata_file, sizex):
    """Load brecons and last computed (l, m) if available."""
    if os.path.exists(save_file) and os.path.exists(metadata_file):
        brecons = np.load(save_file)  # Load previous brecons values
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return brecons, metadata['last_l'], metadata['last_m']
    return np.zeros(sizex, dtype=complex), -1, None  # Start fresh if no saved progress

def reconstqdm(alm, x, y, lmax, map_number, check, verbose=True):
    sizex = x.shape
    reconsdata_folder = os.path.join('reconsdata')
    if not os.path.exists(reconsdata_folder):
        os.makedirs(reconsdata_folder)

    save_file = os.path.join(reconsdata_folder, f"brecons_{'og' if check == 1 else 'pred'}_{map_number}_{lmax}.npy")
    metadata_file = os.path.join(reconsdata_folder, f"brecons_{'og' if check == 1 else 'pred'}_{map_number}_{lmax}_metadata.json")

    brecons, last_l, last_m = load_progress(save_file, metadata_file, sizex)

    total_calculations = (lmax + 1) * (lmax + 1)

    for l in range(last_l + 1, lmax + 1):
        for m in range(-l, l + 1):
            if l == last_l and m <= last_m:
                continue
            if np.isnan(alm[(l, m)]):
                continue  

            ylm = sph_harm(m, l, x, y)
            brecons += alm[(l, m)] * ylm


            save_progress(brecons, save_file, metadata_file, l, m)

    return brecons.real

# Function to convert Carrington map number to month-year string
def get_month_year_from_map_number(map_number):
    base_date = datetime(1853, 11, 9)  # Base date for Carrington rotation 1
    rotation_period_days = 27.2753  # Average Carrington rotation period in days
    days_since_base = (int(map_number) - 1) * rotation_period_days
    map_date = base_date + timedelta(days=days_since_base)
    return map_date.strftime("%B %Y")

def process_carrington_map(carrington_map_number, lmax):
    fits_file = f'fits files/hmi.Synoptic_Mr_small.{carrington_map_number}.fits'

    with fits.open(fits_file) as hdul:
        b = hdul[0].data

    b = np.nan_to_num(b, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

    num_points_theta = b.shape[0]
    num_points_phi = b.shape[1]

    y = np.linspace(0, np.pi, num_points_theta)
    x = np.linspace(0, 2 * np.pi, num_points_phi)
    y, x = np.meshgrid(y, x, indexing='ij')

    # print(f"\nProcess started for Carrington map number {carrington_map_number}...")

    alm = calcalm(lambda theta, phi: b[np.floor(theta * (num_points_theta - 1) / np.pi).astype(int),
                                       np.floor(phi * (num_points_phi - 1) / (2 * np.pi)).astype(int)], 
                  lmax, num_points_theta, num_points_phi, carrington_map_number, check=1)
    
    b_reconstructed_og = reconstqdm(alm, x, y, lmax, carrington_map_number, check=1, verbose=True)  # Pass verbose=True to show progress bar
    
    # print(f"Done reconstructing Carrington map number {carrington_map_number} from OG alm")

    alm2 = calcalm(lambda theta, phi: b[np.floor(theta * (num_points_theta - 1) / np.pi).astype(int),
                                       np.floor(phi * (num_points_phi - 1) / (2 * np.pi)).astype(int)], 
                  lmax, num_points_theta, num_points_phi, carrington_map_number, check=0)
    
    b_reconstructed_pred = reconstqdm(alm2, x, y, lmax, carrington_map_number, check=0, verbose=True)  # Pass verbose=True to show progress bar
    # print(f"Done reconstructing Carrington map number {carrington_map_number} from pred alm")

    b_avg = np.mean(np.abs(b))
    b_recons_og_avg = np.mean(np.abs(b_reconstructed_og))
    b_recons_pred_avg = np.mean(np.abs(b_reconstructed_pred))
    b_avg_values.append(b_avg)
    b_avg_recons_og_values.append(b_recons_og_avg)
    b_avg_recons_pred_values.append(b_recons_pred_avg)
    dates.append(get_month_year_from_map_number(carrington_map_number))
    cr_numbers.append(carrington_map_number)

lmax = 60

print("\n------------------------------------------------------------------------------------------------------------------------")
print("Imported all necessary libraries, compiled successfully.")
print(f"lmax\t\t\t:\t{lmax}")

fits_folder = 'fits files'
fits_files = os.listdir(fits_folder)

plots_dir = "plots/make_animation_3bavg_new_anti_gif"
os.makedirs(plots_dir, exist_ok=True)

lowerbound1 = 2097
numberofmaps = 193  # stops it at 2290
upperbound1 = lowerbound1 + numberofmaps

# frames = []  # List to store images for GIF

for tempupperbound in range(0, numberofmaps):

    b_avg_values = []
    b_avg_recons_og_values = []
    b_avg_recons_pred_values = []
    cr_numbers = []
    dates = []
    
    if tempupperbound in range(23, 32):  # Skip 23 to 31
        continue

    target_cr_numbers = set(range(lowerbound1, lowerbound1 + tempupperbound))

    for fits_filename in fits_files:
        # Extract Carrington Rotation number from filename
        carrington_map_number = int(fits_filename.split('.')[2])

        # Process only the specified CR map numbers
        if carrington_map_number in target_cr_numbers:
            fits_file = os.path.join(fits_folder, fits_filename)

            try:
                with fits.open(fits_file) as hdul:
                    b = hdul[0].data

                b = np.nan_to_num(b, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

                process_carrington_map(carrington_map_number, lmax)

            except Exception as e:
                print(f"Error processing Carrington map number {carrington_map_number}: {e}")
                continue

    # Create the figure and axis
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot b_avg on the left y-axis (blue)
    ax1.plot(cr_numbers, b_avg_values, linestyle='-', color='b', label=r'$B_{avg}$')
    ax1.set_xlabel("Month-Year")
    ax1.set_ylabel("Average Magnetic Flux (Gauss)", color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_title("Solar Magnetic Flux Variation Over Time")

    # Create a twin y-axis
    ax2 = ax1.twinx()

    # Plot the other two b_avg values on the right y-axis (red and green)
    ax2.plot(cr_numbers, b_avg_recons_og_values, linestyle='-', color='r', label=r'$B_{avg}^{og}$')
    ax2.plot(cr_numbers, b_avg_recons_pred_values, linestyle='-', color='g', label=r'$B_{avg}^{pred}$')
    ax2.set_ylabel("Reconstructed Magnetic Flux (Gauss)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Set y-axis limits for both axes from 0 to 14
    ax1.set_ylim(0, 14)
    ax2.set_ylim(0, 14)

    # Set y-ticks explicitly from 0 to 14
    ax1.set_yticks(range(0, 15))
    ax2.set_yticks(range(0, 15))

    # Add legends for both axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Add grid
    ax1.grid()

    # Format x-axis labels
    plt.xticks(ticks=cr_numbers[::20], labels=dates[::20], rotation=45, fontsize=8)
    plt.tight_layout()

    # Save the current frame to memory
    plot_path = os.path.join(plots_dir, f"temp_plot_{tempupperbound}.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)  # Close the plot to free memory
