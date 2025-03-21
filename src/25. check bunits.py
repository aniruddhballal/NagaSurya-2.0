from astropy.io import fits

# Example: Provide your specific FITS file path here
carrington_map_number = 2096  # replace with the desired map number
fits_file = f"fits files/hmi.Synoptic_Mr_small.{carrington_map_number}.fits"

with fits.open(fits_file) as hdul:
    header = hdul[0].header
    bunit = header.get("BUNIT", None)  # Attempt to get the BUNIT keyword

if bunit:
    print(f"The unit of the data in the FITS file is: {bunit}")
else:
    print("No unit information (BUNIT) found in the FITS header.")