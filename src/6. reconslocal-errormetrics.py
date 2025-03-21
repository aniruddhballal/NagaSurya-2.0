import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from astropy.io import fits
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.integrate import simpson
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import json
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics.pairwise import cosine_similarity

IST = timezone(timedelta(hours=5, minutes=30))

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
    if not os.path.exists(values_folder):
        os.makedirs(values_folder)
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

def reconstqdm(alm, x, y, lmax, map_number, check, verbose=True):
    sizex = x.shape
    reconsdata_folder = os.path.join('reconsdata')
    if not os.path.exists(reconsdata_folder):
        os.makedirs(reconsdata_folder)

    save_file = os.path.join(reconsdata_folder, f"brecons_{'og' if check == 1 else 'pred'}_{map_number}_{lmax}.npy")
    metadata_file = os.path.join(reconsdata_folder, f"brecons_{'og' if check == 1 else 'pred'}_{map_number}_{lmax}_metadata.json")

    brecons, last_l, last_m = load_progress(save_file, metadata_file, sizex)

    total_calculations = (lmax + 1) * (lmax + 1)
    progress = tqdm(total=total_calculations, desc=f"Reconstructing map {map_number}", unit="steps", initial=(last_l + 1) * (last_l + 1))

    for l in range(last_l + 1, lmax + 1):
        for m in range(-l, l + 1):
            if l == last_l and m <= last_m:
                continue
            if np.isnan(alm[(l, m)]):
                continue  

            ylm = sph_harm(m, l, x, y)
            brecons += alm[(l, m)] * ylm

            progress.update(1)  # tqdm update

            save_progress(brecons, save_file, metadata_file, l, m)

    progress.close()  # Close tqdm progress bar
    return brecons.real

def plotty(ax, x, y, bvals, type, carrington_map_number, lmax, b_value):
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
        name = f'Reconstructed (OG alms): lmax-{lmax}'
    elif type == 3:
        name = f'Reconstructed (pred alms): lmax-{lmax}'
    ax.set_title(name, fontsize=font_size)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    ax.set_xticklabels([r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=font_size)
    ax.set_yticks(np.linspace(0, np.pi, 5))
    ax.set_yticklabels([r'0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'], fontsize=font_size)
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    # 🔹 Add extra value at the bottom-right corner of the subplot
    ax.text(0.95, 0.05, f'mean(|B|): {b_value:.4f} Gauss', 
            transform=ax.transAxes, fontsize=font_size, 
            verticalalignment='bottom', horizontalalignment='right')

# Function to convert Carrington map number to month-year string
def get_month_year_from_map_number(map_number):
    base_date = datetime(1853, 11, 9)  # Base date for Carrington rotation 1
    rotation_period_days = 27.2753  # Average Carrington rotation period in days
    days_since_base = (int(map_number) - 1) * rotation_period_days
    map_date = base_date + timedelta(days=days_since_base)
    return map_date.strftime("%B %Y")

def compare_maps(original, reconstructed_og, reconstructed_pred):
    """
    Computes similarity metrics between the original CR map, the reconstructed map using OG alm, 
    and the reconstructed map using predicted alm.
    """
    # Compute Mean Absolute Error (MAE)
    mae_og = np.mean(np.abs(original - reconstructed_og))
    mae_pred = np.mean(np.abs(original - reconstructed_pred))
    mae_og_pred = np.mean(np.abs(reconstructed_og - reconstructed_pred))

    # Compute Structural Similarity Index (SSIM)
    ssim_og = ssim(original, reconstructed_og, data_range=original.max() - original.min())
    ssim_pred = ssim(original, reconstructed_pred, data_range=original.max() - original.min())
    ssim_og_pred = ssim(reconstructed_og, reconstructed_pred, data_range=reconstructed_og.max() - reconstructed_og.min())

    # Compute Cosine Similarity
    original_flat = original.flatten().reshape(1, -1)
    reconstructed_og_flat = reconstructed_og.flatten().reshape(1, -1)
    reconstructed_pred_flat = reconstructed_pred.flatten().reshape(1, -1)

    cosine_og = cosine_similarity(original_flat, reconstructed_og_flat)[0, 0]
    cosine_pred = cosine_similarity(original_flat, reconstructed_pred_flat)[0, 0]
    cosine_og_pred = cosine_similarity(reconstructed_og_flat, reconstructed_pred_flat)[0, 0]

    return {
        "MAE (Original vs OG Reconstructed)": mae_og,
        "MAE (Original vs Pred Reconstructed)": mae_pred,
        "MAE (OG Reconstructed vs Pred Reconstructed)": mae_og_pred,
        "SSIM (Original vs OG Reconstructed)": ssim_og,
        "SSIM (Original vs Pred Reconstructed)": ssim_pred,
        "SSIM (OG Reconstructed vs Pred Reconstructed)": ssim_og_pred,
        "Cosine Similarity (Original vs OG Reconstructed)": cosine_og,
        "Cosine Similarity (Original vs Pred Reconstructed)": cosine_pred,
        "Cosine Similarity (OG Reconstructed vs Pred Reconstructed)": cosine_og_pred,
    }

def process_carrington_map(carrington_map_number, lmax):
    fits_file = f'fits files/hmi.Synoptic_Mr_small.{carrington_map_number}.fits'

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

    b_avg = round(np.mean(np.abs(b)), 4)
    plotty(axs[0], x, y, b, 1, carrington_map_number, lmax, b_avg)

    alm = calcalm(lambda theta, phi: b[np.floor(theta * (num_points_theta - 1) / np.pi).astype(int),
                                       np.floor(phi * (num_points_phi - 1) / (2 * np.pi)).astype(int)], 
                  lmax, num_points_theta, num_points_phi, carrington_map_number, check=1)

    # print the alm[(0,0)] to check if its getting the right set of alm values or not
    # print("alm_og[(0,0)]: " + str(alm.get((0, 0), "alm[(0,0)] not computed")))
    
    b_reconstructed_og = reconstqdm(alm, x, y, lmax, carrington_map_number, check=1, verbose=True)  # Pass verbose=True to show progress bar
    
    og_recons_dir = "og reconstructed magnetic fields"
    # Define the output CSV file path
    csv_filename = os.path.join(og_recons_dir, f"magnetic_field_CR{carrington_map_number}.csv")
    # Save the entire 2D array 'b' to a CSV file
    np.savetxt(csv_filename, b_reconstructed_og, delimiter=",", fmt="%.6f")
    print(f"Magnetic field data saved at: {csv_filename}")

    print(f"Done reconstructing Carrington map number {carrington_map_number} from OG alm")

    recons_og_avg = round(np.mean(np.abs(b_reconstructed_og)), 4)
    plotty(axs[1], x, y, b_reconstructed_og, 2, carrington_map_number, lmax, recons_og_avg)

    alm2 = calcalm(lambda theta, phi: b[np.floor(theta * (num_points_theta - 1) / np.pi).astype(int),
                                       np.floor(phi * (num_points_phi - 1) / (2 * np.pi)).astype(int)], 
                  lmax, num_points_theta, num_points_phi, carrington_map_number, check=0)

    # print the alm[(0,0)] to check if its getting the right set of alm values or not
    # print("alm_pred[(0,0)]: " + str(alm2.get((0, 0), "alm[(0,0)] not computed")))
    
    b_reconstructed_pred = reconstqdm(alm2, x, y, lmax, carrington_map_number, check=0, verbose=True)  # Pass verbose=True to show progress bar
    
    pred_recons_dir = "pred reconstructed magnetic fields"
    # Define the output CSV file path
    csv_filename = os.path.join(pred_recons_dir, f"magnetic_field_CR{carrington_map_number}.csv")
    # Save the entire 2D array 'b' to a CSV file
    np.savetxt(csv_filename, b_reconstructed_pred, delimiter=",", fmt="%.6f")
    print(f"Magnetic field data saved at: {csv_filename}")
    
    print(f"Done reconstructing Carrington map number {carrington_map_number} from pred alm")

    recons_pred_avg = round(np.mean(np.abs(b_reconstructed_pred)), 4)
    plotty(axs[2], x, y, b_reconstructed_pred, 3, carrington_map_number, lmax, recons_pred_avg)
    
    axs[3].axis('off')

    # find a way to display these values
    # plt.figtext(0.95, 0.05, f"B Avg: {b_avg:.4f}\nRecons OG Avg: {recons_og_avg:.4f}\nRecons Pred Avg: {recons_pred_avg:.4f}\n\n", va='bottom', ha='right')

    # Compute similarity metrics
    similarity_metrics = compare_maps(b, b_reconstructed_og, b_reconstructed_pred)

    # Print results
    for metric, value in similarity_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Display similarity results on the plot
    similarity_text = "\n".join([f"{metric}: {value:.4f}" for metric, value in similarity_metrics.items()])
    axs[3].text(0.5, 0.5, similarity_text, va='center', ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

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

    plt.show()

# print("enter lmax: ")
# lmax = int(input())
lmax = 60

print("\n------------------------------------------------------------------------------------------------------------------------")
print("Imported all necessary libraries, compiled successfully.")
print(f"lmax\t\t\t:\t{lmax}")
# input("Press Enter to confirm these values and start the process:")

fits_folder = 'fits files'
fits_files = os.listdir(fits_folder)

for fits_filename in fits_files:
    if fits_filename == 'hmi.Synoptic_Mr_small.2097.fits':
        carrington_map_number = int(fits_filename.split('.')[2])

        fits_file = os.path.join(fits_folder, fits_filename)

        try:
            with fits.open(fits_file) as hdul:
                b = hdul[0].data

            b = np.nan_to_num(b, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

            process_carrington_map(carrington_map_number, lmax)

        except Exception as e:
            print(f"Error processing Carrington map number {carrington_map_number}: {e}")
            continue