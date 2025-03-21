import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime, timedelta, timezone
import winsound
from tqdm import tqdm
from astropy.io import fits
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.integrate import simpson
import shutil

IST = timezone(timedelta(hours=5, minutes=30))

def read_alm_from_csv(csv_filename):
    alm = {}
    if os.path.exists(csv_filename):
        with open(csv_filename, 'r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                l = int(row['l'])
                m = int(row['m'])
                alm[(l, m)] = complex(row['alm'])  # string to complex
    return alm

def write_alm_to_csv(csv_filename, alm):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['l', 'm', 'alm'])
        for (l, m), value in alm.items():
            writer.writerow([l, m, value])

def calcalm(b_func, lmax, num_points_theta, num_points_phi, carrington_map_number, verbose=True):
    alm = {}

    values_folder = "alm values"
    if not os.path.exists(values_folder):
        os.makedirs(values_folder)
    values_filename = os.path.join(values_folder, f'values_{carrington_map_number}.csv')
    alm.update(read_alm_from_csv(values_filename))

    def integrand(theta, phi, l, m):
        ylm = sph_harm(m, l, phi, theta)
        return b_func(theta, phi) * np.conj(ylm) * np.sin(theta)

    total_calculations = (lmax + 1) * (lmax + 1)
    progress_bar = tqdm(total=total_calculations, desc=f"Calculating alm (Carrington map {carrington_map_number})") if verbose else None

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

def recons(alm, x, y, lmax, verbose=True):
    sizex = x.shape
    brecons = np.zeros(sizex, dtype=complex)

    total_calculations = (lmax + 1) * (lmax + 1)
    progress_bar = tqdm(total=total_calculations, desc="Reconstructing data") if verbose else None

    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            if np.isnan(alm[(l, m)]):
                continue
            ylm = sph_harm(m, l, x, y)
            brecons += alm[(l, m)] * ylm
            if verbose:
                progress_bar.update(1)

    if verbose:
        progress_bar.close()

    return brecons.real

def plotty(ax, x, y, bvals, type, carrington_map_number):
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

    month_year = carrington_map_dates.get(carrington_map_number, "Unknown Date")

    name = ''
    if type == 1:
        name = f'{month_year} - CR map: {carrington_map_number}'
    elif type == 2:
        name = f'Reconstructed (S 1/3 Rule)'
    elif type == 3:
        name = 'Delta/Error'
    ax.set_title(name, fontsize=font_size)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=font_size)
    ax.set_yticks(np.linspace(0, np.pi, 5))
    ax.set_yticklabels([r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)

def process_carrington_map(carrington_map_number, lmax):
    fits_file = f'C:/Users/aniru/pyproj/my_env1/we goin solar/03Jul/fits_files/hmi.Synoptic_Mr_small.{carrington_map_number}.fits'
    # f'E:/Project-NagaSurya/03Jul/sample fits file/hmi.Synoptic_Mr_small.{carrington_map_number}.fits'

    with fits.open(fits_file) as hdul:
        b = hdul[0].data

    b = np.nan_to_num(b, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

    sig = 2
    b = gaussian_filter(b, sigma=sig)

    num_points_theta = b.shape[0]
    num_points_phi = b.shape[1]

    y = np.linspace(0, np.pi, num_points_theta)
    x = np.linspace(0, 2 * np.pi, num_points_phi)
    y, x = np.meshgrid(y, x, indexing='ij')

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()

    start_ist = datetime.now(IST)
    start_ist_str = start_ist.strftime("%Y-%m-%d %H:%M:%S %Z")

    print(f"\nProcess started for Carrington map number {carrington_map_number}...\nIST now: {start_ist_str}")

    plotty(axs[0], x, y, b, 1, carrington_map_number)

    alm = calcalm(lambda theta, phi: b[np.floor(theta * (num_points_theta - 1) / np.pi).astype(int),
                                       np.floor(phi * (num_points_phi - 1) / (2 * np.pi)).astype(int)], 
                  lmax, num_points_theta, num_points_phi, carrington_map_number)
    
    beepbeep(750,600)

    b_reconstructed = recons(alm, x, y, lmax, verbose=True)  # Pass verbose=True to show progress bar
    print(f"Done reconstructing Carrington map number {carrington_map_number}")

    plotty(axs[1], x, y, b_reconstructed, 2, carrington_map_number)

    delta = b - b_reconstructed
    plotty(axs[2], x, y, delta, 3, carrington_map_number)

    max_delta = round(np.max(delta), 4)
    min_delta = round(np.min(delta), 4)
    avg_abs_delta = round(np.mean(np.abs(delta)), 4)
    recons_avg = round(np.mean(np.abs(b_reconstructed)), 4)
    b_avg = round(np.mean(np.abs(b)), 4)
    ndiff = round(np.abs(recons_avg - b_avg), 4)

    errors_folder = "error metrics"
    if not os.path.exists(errors_folder):
        os.makedirs(errors_folder)
    error_filename = os.path.join(errors_folder, f'error_{carrington_map_number}.csv')
    with open(error_filename, 'w', newline='') as btxt:
        writer = csv.writer(btxt)
        writer.writerow(['Max Delta', 'Min Delta', 'Avg Abs Delta', 'B Avg', 'Recons Avg', 'Net Diff'])
        writer.writerow([max_delta, min_delta, avg_abs_delta, b_avg, recons_avg, ndiff])
    
    axs[3].axis('off')

    plt.figtext(0.95, 0.05, f"B Avg: {b_avg:.4f}\nRecons Avg: {recons_avg:.4f}\nNet Diff: {ndiff:.4f}\n\n", va='bottom', ha='right')

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.98, top=0.6, bottom=0, wspace=0.25, hspace=0.2)

    for ax in axs:
        ax.set_aspect(1.2 / 1.0, adjustable='box')

    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    plot_filename = os.path.join(plots_folder, f'plot_{carrington_map_number}.png')
    plt.savefig(plot_filename, bbox_inches='tight')

    print("Saved the plot.")

    beepbeep(1000,600)

def beepbeep(f, d):
    winsound.Beep(f, d)

carrington_map_dates = {
    2096: "April 2010",
}

lmax = 85
method = "Simpson's 1/3 Rule"

print("\n------------------------------------------------------------------------------------------------------------------------")
print("Imported all necessary libraries, compiled successfully.")
print(f"lmax\t\t\t:\t{lmax}")
print("Method of integration\t:\t" + method)
input("Press Enter to confirm these values and start the process:")

beepbeep(500, 1300)

fits_folder = f'sample fits file'
fits_files = os.listdir(fits_folder)

done_folder = os.path.join(fits_folder, 'done')
if not os.path.exists(done_folder):
    os.makedirs(done_folder)

for fits_filename in fits_files:
    if fits_filename.startswith('hmi.Synoptic_Mr_small.') and fits_filename.endswith('.fits'):
        carrington_map_number = int(fits_filename.split('.')[2])

        fits_file = os.path.join(fits_folder, fits_filename)

        try:
            with fits.open(fits_file) as hdul:
                b = hdul[0].data

            b = np.nan_to_num(b, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

            process_carrington_map(carrington_map_number, lmax)

            shutil.move(fits_file, os.path.join(done_folder, fits_filename))

        except Exception as e:
            print(f"Error processing Carrington map number {carrington_map_number}: {e}")
            continue