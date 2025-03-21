import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Function to load and visualize the FITS file
def load_and_visualize_fits(file_path):
    # Open the FITS file
    with fits.open(file_path) as hdul:
        hdul.info()  # Print FITS file structure
        data = hdul[0].data  # Extract magnetogram data

    # Handle missing values (replace NaNs with 0)
    data = np.nan_to_num(data)

    # Plot the synoptic magnetogram
    plt.figure(figsize=(10, 5))
    plt.imshow(data, cmap='bwr', origin='lower', aspect='auto')
    plt.colorbar(label="Magnetic Field Strength (Gauss)")
    plt.title(f"Magnetic Field Map: {file_path.split('.')[-1]}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

    return data

# Example usage
file_path = "fits_files/hmi.Synoptic_Mr_small.2267.fits"  # Change this to your actual file path
magnetogram_data = load_and_visualize_fits(file_path)