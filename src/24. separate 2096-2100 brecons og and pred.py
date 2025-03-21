import os
import shutil
import re

# Define source and destination folders
source_folder = "reconsdata"
destination_folder = "recons_og"

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Regex pattern to match filenames like "brecons_og_xxxx_yy_metadata.json", "brecons_og_xxxx_y_metadata.json", "brecons_og_xxxx_0_metadata.json", or "brecons_og_xxxx_0.npy" with xxxx in range 2096-2100
pattern1 = re.compile(r"brecons_og_(209[6-9]|2100)_[0-9]+_(metadata\.json|npy)$")
pattern2 = re.compile(r"brecons_og_(209[6-9]|2100)_[0-9]+\.npy$")

# Iterate through files in the source folder
for filename in os.listdir(source_folder):
    if pattern1.match(filename) or pattern2.match(filename):
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(destination_folder, filename)
        shutil.copy2(source_path, destination_path)  # Preserves metadata like timestamps
        print(f"Copied: {filename}")

print("✅ Copying complete.")