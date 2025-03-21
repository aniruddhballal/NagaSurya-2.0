import numpy as np
from scipy.special import sph_harm
import csv
import os
from astropy.io import fits
from scipy.integrate import simpson
import json
import matplotlib.pyplot as plt
from rich.progress import Progress
# from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap


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

def calc_recons(map_number, lmax, alm, x, y, b_avg):
    sizex = x.shape
    reconsdata_folder = os.path.join('recons_og')

    save_file = os.path.join(reconsdata_folder, f"brecons_og_{map_number}_{lmax}.npy")
    metadata_file = os.path.join(reconsdata_folder, f"brecons_og_{map_number}_{lmax}_metadata.json")

    brecons, last_l, last_m = load_progress(save_file, metadata_file, sizex)

    for l in range(last_l + 1, lmax + 1):
        for m in range(-l, l + 1):
            if l == last_l and m <= last_m:
                continue
            if np.isnan(alm[(l, m)]):
                continue  

            ylm = sph_harm(m, l, x, y)
            brecons += alm[(l, m)] * ylm

            save_progress(brecons, save_file, metadata_file, l, m)

    b_reconstructed_og = brecons.real
    b_recons_og_avg = np.mean(np.abs(b_reconstructed_og))
    frac = b_recons_og_avg/b_avg
    frac_values.append(frac)
    l_values.append(lmax)

def plotty200(ax, x, y, bvals):
    font_size = 20
    contnum = 50

    vmin = np.min(bvals)
    vmax = np.max(bvals)

    clevels = np.linspace(-200, 200, contnum + 1)
    
    # Create custom colormap
    colors = ["blue", "white", "red"]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
    
    bvals = np.clip(bvals, -200, 200)

    contourf_plot = ax.contourf(x, y, bvals, levels=clevels, cmap=cmap, vmin=-200, vmax=200)

    cbar = plt.colorbar(contourf_plot, ax=ax)
    cbar.set_label('Gauss (G)', fontsize=20)  # Increase the font size here 
    cbar.set_ticks([-200, 200])
    cbar.set_ticklabels([f'{-200}', f'{200}'], fontsize=font_size)
    ax.set_xlabel(r'$\phi$', fontsize=font_size)
    ax.set_ylabel(r'$\theta$', fontsize=font_size)

    name = f'saturated to +/-200 (og range: {vmin:.2f} to {vmax:.2f})'
    
    ax.set_title(name, fontsize=font_size)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=font_size)
    ax.set_yticks(np.linspace(0, np.pi, 5))
    ax.set_yticklabels([r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.grid(True)

lmax = 85

print("\n------------------------------------------------------------------------------------------------------------------------")
print("Imported all necessary libraries, compiled successfully.")
print(f"lmax\t\t\t:\t{lmax}")

fits_folder = 'fits files'
fits_files = os.listdir(fits_folder)

plots_dir = "plots/fractional_b_varying_with_l"
os.makedirs(plots_dir, exist_ok=True)

# target_cr_numbers = set(range(2096, 2101)) | {2226, 2253, 2265}# | set(range(2128, 2291))  # 2097-2118 and 2128-2290
target_cr_numbers = set(range(2097, 2098))# | {2226, 2253, 2265}# | set(range(2128, 2291))  # 2097-2118 and 2128-2290


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
        with Progress() as progress:
            task = progress.add_task(f"[cyan]Processing b_recons variation for CR {carrington_map_number}...", total=lmax + 1)
            
            for l in range(lmax + 1):
                try:
                    calc_recons(carrington_map_number, l, alm, x, y, b_avg)
                    progress.update(task, advance=1)

                except Exception as e:
                    print(f"Error processing Carrington map number {carrington_map_number}: {e}")
                    continue

        fig, axs = plt.subplots(1, 2, figsize=(40, 10), gridspec_kw={'width_ratios': [1, 1]})  # Two plots side by side

        # First Plot - Fractional B values
        axs[0].plot(l_values, frac_values, linestyle='-', color='b', label='fractional brecons values')

        # Highlight max values with different markers
        sorted_indices = np.argsort(frac_values)[-5:]
        top_max_values = [frac_values[i] for i in sorted_indices]
        top_l_values = [l_values[i] for i in sorted_indices]

        colors = ['#FF5733', '#33FF57', '#5733FF', '#FF33A1', '#FFD700']
        markers = ['o', 's', '^', '*', 'D']

        for i in range(5):
            axs[0].scatter(top_l_values[4 - i], top_max_values[4 - i], color=colors[i], marker=markers[i], zorder=3, label=f'{top_max_values[4 - i]:.4f}')

        axs[0].set_xlabel('l Values')
        axs[0].set_ylabel(f'Fraction of B_Avg ({b_avg:.2f} G)')
        axs[0].set_title(f'Carrington Map {carrington_map_number} - Variation of fractional B reconstructed')
        axs[0].set_xticks(np.arange(0, max(l_values) + 1, 10))
        axs[0].grid(True)
        axs[0].legend()

        # Second Plot - Using plotty200
        plotty200(axs[1], x, y, b)  # Assuming plotty200 correctly plots on axs[1]

        # Save and show
        plot_path = os.path.join(plots_dir, f"{carrington_map_number}_combined_plot.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
        plt.show()
