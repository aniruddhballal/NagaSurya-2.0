import numpy as np
from scipy.special import sph_harm
import csv, os, numpy as np
from scipy.integrate import simpson
from astropy.io import fits
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import json
from tqdm import tqdm
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

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

def calcalm(b_func, lmax, num_points_theta, num_points_phi, carrington_map_number, check, verbose=True):
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

    total_calculations = (lmax + 1) * (lmax + 1)
    progress_bar = tqdm(total=total_calculations, desc=f"(lmax: {lmax} - Calculating alm (Carrington map {carrington_map_number})") if verbose else None

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
            if verbose:
                progress_bar.update(1)

    if verbose:
        progress_bar.close()
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

def recons(alm, x, y, lmax, map_number, check, verbose=True):
    sizex = x.shape
    reconsdata_folder = os.path.join('reconsdata')
    if not os.path.exists(reconsdata_folder):
        os.makedirs(reconsdata_folder)
    save_file = ""
    metadata_file = ""
    if check == 1:
        save_file = os.path.join(reconsdata_folder, f"brecons_og_{map_number}_{lmax}.npy")  # File to save reconstruction progress
        metadata_file = os.path.join(reconsdata_folder, f"brecons_og_{map_number}_{lmax}_metadata.json")  # Metadata file for tracking progress
    else:
        save_file = os.path.join(reconsdata_folder, f"brecons_pred_{map_number}_{lmax}.npy")  # File to save reconstruction progress
        metadata_file = os.path.join(reconsdata_folder, f"brecons_pred_{map_number}_{lmax}_metadata.json")  # Metadata file for tracking progress

    # Load saved progress if available
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
        task = progress.add_task(f"[red]Reconstructing map {map_number}", total=total_calculations)
        progress.update(task, completed=(last_l + 1) * (last_l + 1))  # Resume progress
        progress.start()

    for l in range(last_l + 1, lmax + 1):  # Resume from last computed l
        for m in range(-l, l + 1):
            if l == last_l and m <= last_m:  # Skip already computed values
                continue

            if np.isnan(alm[(l, m)]):
                continue  # Skip if alm is NaN

            ylm = sph_harm(m, l, x, y)
            brecons += alm[(l, m)] * ylm

            if verbose:
                progress.update(task, advance=1)

            # Save progress after each (l, m) computation
            save_progress(brecons, save_file, metadata_file, l, m)

    if verbose:
        progress.stop()
    return brecons.real  # Final reconstructed output

# Function to convert Carrington map number to month-year string
def get_month_year_from_map_number(map_number):
    base_date = datetime(1853, 11, 9)  # Base date for Carrington rotation 1
    rotation_period_days = 27.2753  # Average Carrington rotation period in days
    days_since_base = (int(map_number) - 1) * rotation_period_days
    map_date = base_date + timedelta(days=days_since_base)
    return map_date.strftime("%B %Y")

def plotty(ax, x, y, bvals, type, carrington_map_number, lmax):
    font_size = 12
    contnum = 50

    vmin = np.min(bvals)
    vmax = np.max(bvals)
    absmax = max(abs(vmin), abs(vmax))
    clevels = np.linspace(-absmax, absmax, contnum + 1)
    
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    
    contourf_plot = ax.contourf(x, y, bvals, levels=clevels, cmap=cmap, vmin=-absmax, vmax=absmax)

    cbar = plt.colorbar(contourf_plot, ax=ax, label='Gauss (G)')
    cbar.set_ticks([-absmax, absmax])
    cbar.set_ticklabels([f'{-absmax:.2f}', f'{absmax:.2f}'], fontsize=font_size)
    ax.set_xlabel(r'$\phi$', fontsize=font_size)
    ax.set_ylabel(r'$\theta$', fontsize=font_size)

    # month_year = carrington_map_dates.get(carrington_map_number, "Unknown Date")
    month_year = get_month_year_from_map_number(carrington_map_number)

    name = ''
    if type == 1:
        name = f'{month_year} - CR map: {carrington_map_number}'
    elif type == 2:
        name = f'{month_year} - CR map: {carrington_map_number}: Gaussian 1'
    elif type == 3:
        name = f'{month_year} - CR map: {carrington_map_number}: Gaussian 2'
    elif type == 4:
        name = f'{month_year} - CR map: {carrington_map_number}: Gaussian 3'
    ax.set_title(name, fontsize=font_size)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=font_size)
    ax.set_yticks(np.linspace(0, np.pi, 5))
    ax.set_yticklabels([r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)

def plotty200(ax, x, y, bvals, type, carrington_map_number, lmax):
    font_size = 12
    contnum = 50

    bvals = np.clip(bvals, None, 200)  # Replaces any value >= 200 with 200

    vmin = np.min(bvals)
    vmax = np.max(bvals)
    absmax = max(abs(vmin), abs(vmax))
    clevels = np.linspace(-absmax, absmax, contnum + 1)
    
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    
    contourf_plot = ax.contourf(x, y, bvals, levels=clevels, cmap=cmap, vmin=-200, vmax=200)

    cbar = plt.colorbar(contourf_plot, ax=ax, label='Gauss (G)')
    cbar.set_ticks([-absmax, absmax])
    cbar.set_ticklabels([f'{-absmax:.2f}', f'{absmax:.2f}'], fontsize=font_size)
    ax.set_xlabel(r'$\phi$', fontsize=font_size)
    ax.set_ylabel(r'$\theta$', fontsize=font_size)

    # month_year = carrington_map_dates.get(carrington_map_number, "Unknown Date")
    month_year = get_month_year_from_map_number(carrington_map_number)

    name = ''
    if type == 1:
        name = f'{month_year} - CR map: {carrington_map_number}'
    elif type == 2:
        name = f'{month_year} - CR map: {carrington_map_number}: Gaussian 1'
    elif type == 3:
        name = f'{month_year} - CR map: {carrington_map_number}: Gaussian 2'
    elif type == 4:
        name = f'{month_year} - CR map: {carrington_map_number}: Gaussian 3'
    ax.set_title(name, fontsize=font_size)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=font_size)
    ax.set_yticks(np.linspace(0, np.pi, 5))
    ax.set_yticklabels([r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)

def process_carrington_map(carrington_map_number, lmax):
    fits_file = f'fits files/hmi.Synoptic_Mr_small.{carrington_map_number}.fits'

    with fits.open(fits_file) as hdul:
        b = hdul[0].data

    b = np.nan_to_num(b, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

    b_gauss_1 = gaussian_filter(b, sigma=1)
    b_gauss_2 = gaussian_filter(b, sigma=2)
    b_gauss_3 = gaussian_filter(b, sigma=3)

    b_gauss_1 = np.clip(b_gauss_1, None, 200)
    b_gauss_2 = np.clip(b_gauss_2, None, 200)
    b_gauss_3 = np.clip(b_gauss_3, None, 200)

    num_points_theta = b.shape[0]
    num_points_phi = b.shape[1]

    y = np.linspace(0, np.pi, num_points_theta)
    x = np.linspace(0, 2 * np.pi, num_points_phi)
    y, x = np.meshgrid(y, x, indexing='ij')

    print(f"\nProcess started for Carrington map number {carrington_map_number}...")

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()

    plotty(axs[0], x, y, b, 1, carrington_map_number, lmax)
    plotty(axs[1], x, y, b_gauss_1, 2, carrington_map_number, lmax)
    plotty(axs[2], x, y, b_gauss_2, 3, carrington_map_number, lmax)
    plotty(axs[3], x, y, b_gauss_3, 4, carrington_map_number, lmax)


    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.98, top=0.6, bottom=0, wspace=0.25, hspace=0.2)

    for ax in axs:
        ax.set_aspect(1.2 / 1.0, adjustable='box')

    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    plot_filename = os.path.join(plots_folder, f'plot_{carrington_map_number}_{lmax}.png')
    plt.savefig(plot_filename, bbox_inches='tight')

    print("Saved the plot.")

    fig2, axs2 = plt.subplots(2, 2, figsize=(15, 12))
    axs2 = axs2.flatten()

    plotty200(axs2[0], x, y, b, 1, carrington_map_number, lmax)
    plotty200(axs2[1], x, y, b_gauss_1, 2, carrington_map_number, lmax)
    plotty200(axs2[2], x, y, b_gauss_2, 3, carrington_map_number, lmax)
    plotty200(axs2[3], x, y, b_gauss_3, 4, carrington_map_number, lmax)

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.98, top=0.6, bottom=0, wspace=0.25, hspace=0.2)

    for ax in axs2:
        ax.set_aspect(1.2 / 1.0, adjustable='box')

    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    plot_filename = os.path.join(plots_folder, f'plot_{carrington_map_number}_{lmax}_saturated.png')
    plt.savefig(plot_filename, bbox_inches='tight')

    print("Saved the plot.")

    alm = calcalm(lambda theta, phi: b[np.floor(theta * (num_points_theta - 1) / np.pi).astype(int),
                                       np.floor(phi * (num_points_phi - 1) / (2 * np.pi)).astype(int)], 
                  lmax, num_points_theta, num_points_phi, carrington_map_number, check=1)
    
    # b_reconstructed_og = recons(alm, x, y, lmax, carrington_map_number, check=1, verbose=True)  # Pass verbose=True to show progress bar
    # print(f"Done reconstructing Carrington map number {carrington_map_number} from OG alm")

    '''alm2 = calcalm(lambda theta, phi: b[np.floor(theta * (num_points_theta - 1) / np.pi).astype(int),
                                       np.floor(phi * (num_points_phi - 1) / (2 * np.pi)).astype(int)], 
                  lmax, num_points_theta, num_points_phi, carrington_map_number, check=0)
    
    b_reconstructed_pred = recons(alm2, x, y, lmax, carrington_map_number, check=0, verbose=True)  # Pass verbose=True to show progress bar
    print(f"Done reconstructing Carrington map number {carrington_map_number} from pred alm")'''

lmax = 5

print("\n------------------------------------------------------------------------------------------------------------------------")
print("Imported all necessary libraries, compiled successfully.")
print(f"lmax\t\t\t:\t{lmax}")

fits_folder = 'fits files'
fits_files = os.listdir(fits_folder)

# Define the specific CR map numbers to process
target_cr_numbers = {2096, 2097, 2098}

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