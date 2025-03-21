import os
import time
import winsound
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from selenium.common.exceptions import TimeoutException, WebDriverException

# Define Firefox options
options = Options()
options.headless = False  # Set to True if you don't want to see the Firefox browser window

# Set the download directory
download_dir = r"fits file"  # Replace with your desired path

# Create the directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

# Set Firefox options for download
options.set_preference("browser.download.folderList", 2)  # Use custom download location
options.set_preference("browser.download.manager.showWhenStarting", False)
options.set_preference("browser.download.dir", download_dir)
options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/octet-stream")  # MIME type for FITS files

# Set up the WebDriver service
service = Service(executable_path=r'C:\Users\aniru\pyproj\my_env1\we goin solar\03Jul\gecko\geckodriver-v0.34.0-win64\geckodriver.exe')

# Initialize the Firefox WebDriver with the options
driver = webdriver.Firefox(service=service, options=options)

def download_fits_file(driver, carrington_map):
    try:
        # Navigate to the Stanford HMI synoptic map page
        driver.get("http://hmi.stanford.edu/data/synoptic.html")

        # Find the input field for Carrington map number
        input_elem = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.NAME, "cr")))

        # Clear the input field (if needed) and enter the current Carrington map number
        input_elem.clear()
        input_elem.send_keys(str(carrington_map))

        # Find and click the "Retrieve charts" button
        retrieve_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//input[@type='submit' and @value='Retrieve Charts']")))
        retrieve_button.click()

        # Wait for the next page to load
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//table")))

        # Look for the FITS file link and click on it to download
        file_link = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, f"//a[contains(@href, 'hmi.Synoptic_Mr_small.{carrington_map}.fits')]")))
        file_link.click()

        # Wait for the download to complete (you might need to adjust the wait time based on file size and network speed)
        time.sleep(5)  # Adjust sleep time as needed

        # Optional: Rename downloaded file if needed
        # new_filename = f"CR{carrington_map}.fits"
        # os.rename(os.path.join(download_dir, f"hmi.Synoptic_Mr_small.{carrington_map}.fits"),
        #           os.path.join(download_dir, new_filename))

        return True  # Return True if download was successful

    except (TimeoutException, WebDriverException) as e:
        print(f"Download for Carrington map {carrington_map} failed: {e}")
        return False  # Return False if download failed

    except Exception as e:
        print(f"Unexpected error for Carrington map {carrington_map}: {e}")
        return False  # Return False for any unexpected error


winsound.Beep(1000,600)
# Loop through Carrington map numbers from 2285 to 2294
for carrington_map in range(2285, 2294):
    max_attempts = 3
    attempt = 1
    while attempt <= max_attempts:
        if download_fits_file(driver, carrington_map):
            print(f"Download for Carrington map {carrington_map} successful.")
            break  # Break out of the retry loop if download is successful
        else:
            print(f"Retrying download for Carrington map {carrington_map}. Attempt {attempt}...")
            attempt += 1
            time.sleep(5)  # Wait for a few seconds before retrying

    if attempt > max_attempts:
        print(f"Download for Carrington map {carrington_map} failed after {max_attempts} attempts.")

# Close the WebDriver session
driver.quit()
winsound.Beep(1000,600)