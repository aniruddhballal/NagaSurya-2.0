import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Function to plot solar magnetic field data inside a sine wave background
def plot_solar_magnetic_field(fits_file, save_path='plots/simpfitsoverlap.png', inset_position=[0.5, 0.5, 0.4, 0.4]):
    # Open the FITS file and extract data
    with fits.open(fits_file) as hdul:
        magnetic_field_data = hdul[0].data  # Magnetic field data is usually in the first HDU

    # Handle NaN or infinite values (common in FITS data)
    magnetic_field_data = np.nan_to_num(magnetic_field_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Create sine wave background
    x = np.linspace(0, 10, 500)
    y = np.linspace(0, 10, 500)
    X, Y = np.meshgrid(x, y)
    sine_wave = np.sin(X) * np.cos(Y)

    fig, ax = plt.subplots(figsize=(10, 8))
    sine_map = ax.imshow(sine_wave, cmap='viridis', origin='lower')
    plt.colorbar(sine_map, ax=ax, label='Sine Wave Amplitude')

    # Create inset for solar magnetic field map
    inset_ax = fig.add_axes(inset_position)  # [left, bottom, width, height]
    solar_map = inset_ax.imshow(magnetic_field_data, cmap='seismic', origin='lower')

    # Remove solar map ticks
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    # Add colorbar for the solar map with matching height
    cb = plt.colorbar(solar_map, ax=inset_ax, label='Magnetic Field Strength (Gauss)', 
                      location='left', fraction=0.05, shrink=0.6, pad=0.05)
    cb.ax.tick_params(labelsize=8)

    # Titles and labels
    ax.set_title("Sine Wave with Solar Magnetic Field Map Overlay")
    inset_ax.set_title("Solar Magnetic Field Map", fontsize=8)

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()

# Example usage
fits_file = 'fits files/hmi.Synoptic_Mr_small.2096.fits'  # Replace with your FITS file path
plot_solar_magnetic_field(fits_file, inset_position=[0.38, 0.055, 0.34, 0.4])