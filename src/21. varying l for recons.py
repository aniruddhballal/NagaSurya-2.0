import numpy as np
from scipy.special import sph_harm
import csv
import os
from astropy.io import fits
from scipy.integrate import simpson
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
# from scipy.ndimage import gaussian_filter

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

def calcalm(b_func, lmax, num_points_theta, num_points_phi, carrington_map_number):
    alm = {}
    values_folder = "alm_values_split"
    if not os.path.exists(values_folder):
        os.makedirs(values_folder)
    values_filename = os.path.join(values_folder, f'values_{carrington_map_number}.csv')
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
    if verbose:
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn()
        )
        task = progress.add_task(f"[red]Reconstructing map {map_number} upto l={lmax}", total=total_calculations)
        progress.update(task, completed=(last_l + 1) * (last_l + 1))  # Resume progress
        progress.start()

    for l in range(last_l + 1, lmax + 1):
        for m in range(-l, l + 1):
            if l == last_l and m <= last_m:
                continue
            if np.isnan(alm[(l, m)]):
                continue  

            ylm = sph_harm(m, l, x, y)
            brecons += alm[(l, m)] * ylm

            if verbose:
                progress.update(task, advance=1)

            save_progress(brecons, save_file, metadata_file, l, m)

    if verbose:
        progress.stop()
    return brecons.real

# Function to convert Carrington map number to month-year string
def get_month_year_from_map_number(map_number):
    base_date = datetime(1853, 11, 9)  # Base date for Carrington rotation 1
    rotation_period_days = 27.2753  # Average Carrington rotation period in days
    days_since_base = (int(map_number) - 1) * rotation_period_days
    map_date = base_date + timedelta(days=days_since_base)
    return map_date.strftime("%B %Y")

def process_carrington_map(carrington_map_number, lmax, alm, x, y, b_avg):
    b_reconstructed_og = reconstqdm(alm, x, y, lmax, carrington_map_number, check=1, verbose=True)  # Pass verbose=True to show progress bar
    b_recons_og_avg = np.mean(np.abs(b_reconstructed_og))
    frac = b_recons_og_avg/b_avg
    frac_values.append(frac)
    l_values.append(lmax)

lmax = 85

print("\n------------------------------------------------------------------------------------------------------------------------")
print("Imported all necessary libraries, compiled successfully.")
print(f"lmax\t\t\t:\t{lmax}")

fits_folder = 'fits files'
fits_files = os.listdir(fits_folder)

plots_dir = "plots/fractional_b_varying_with_l"
os.makedirs(plots_dir, exist_ok=True)

target_cr_numbers = set(range(2096, 2101))# | set(range(2128, 2291))  # 2097-2118 and 2128-2290

for fits_filename in fits_files:
    # Extract Carrington Rotation number from filename
    carrington_map_number = int(fits_filename.split('.')[2])

    # Process only the specified CR map numbers
    if carrington_map_number in target_cr_numbers:
        fits_file = os.path.join(fits_folder, fits_filename)
        l_values = []
        frac_values = []
        fits_file = f'fits files/hmi.Synoptic_Mr_small.{carrington_map_number}.fits'
        with fits.open(fits_file) as hdul:
            b = hdul[0].data

        b = np.nan_to_num(b, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

        # Apply Gaussian smoothing
        # sig = 3
        # b = gaussian_filter(bgauss, sigma=sig)  # Adjust sigma as needed for smoothing

        num_points_theta = b.shape[0]
        num_points_phi = b.shape[1]

        y = np.linspace(0, np.pi, num_points_theta)
        x = np.linspace(0, 2 * np.pi, num_points_phi)
        y, x = np.meshgrid(y, x, indexing='ij')

        alm = calcalm(lambda theta, phi: b[np.floor(theta * (num_points_theta - 1) / np.pi).astype(int),
                                        np.floor(phi * (num_points_phi - 1) / (2 * np.pi)).astype(int)], 
                    lmax, num_points_theta, num_points_phi, carrington_map_number)

        b_avg = np.mean(np.abs(b))
        for l in range (lmax+1):
            try:
                process_carrington_map(carrington_map_number, l, alm, x, y, b_avg)
                print(f'done processing: {carrington_map_number}')

            except Exception as e:
                print(f"Error processing Carrington map number {carrington_map_number}: {e}")
                continue

        # Plotting
        plt.figure(figsize=(8, 5))
        plt.plot(l_values, frac_values, linestyle='-', color='b', label='Fraction Values')

        # Identify top 3 maximum values and their corresponding l_values
        sorted_indices = np.argsort(frac_values)[-5:]  # Indices of top 3 max values
        top_max_values = [frac_values[i] for i in sorted_indices]
        top_l_values = [l_values[i] for i in sorted_indices]

        # Highlight max values with different markers
        plt.scatter(top_l_values[4], top_max_values[4], color='#FF5733', marker='o', zorder=3, label=f'{top_max_values[4]:.4f}')
        plt.scatter(top_l_values[3], top_max_values[3], color='#33FF57', marker='s', zorder=3, label=f'{top_max_values[3]:.4f}')
        plt.scatter(top_l_values[2], top_max_values[2], color='#5733FF', marker='^', zorder=3, label=f'{top_max_values[2]:.4f}')
        plt.scatter(top_l_values[1], top_max_values[1], color='#FF33A1', marker='*', zorder=3, label=f'{top_max_values[1]:.4f}')
        plt.scatter(top_l_values[0], top_max_values[0], color='#FFD700', marker='D', zorder=3, label=f'{top_max_values[0]:.4f}')

        # Adding labels and title
        plt.xlabel('l Values')
        plt.ylabel('Fraction Values')
        plt.title(f'Carrington Map {carrington_map_number} - Variation of fractional B reconstructed')

        # Customizing X-axis ticks
        plt.xticks(np.arange(0, max(l_values) + 1, 10))  # Ticks starting from 0 with steps of 10

        plt.grid(True)
        plt.legend()

        plot_path = os.path.join(plots_dir, f"{carrington_map_number}_frac_b_upto_lmax{lmax}.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)

        # Display the plot
        plt.show()