import os
import pandas as pd

# Define the folder containing the CSV files
folder_path = "alm_values"
output_folder = "alm_values_split"

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the range of Carrington Rotation numbers
start_cr, end_cr = 2096, 2285

for cr_number in range(start_cr, end_cr + 1):
    file_name = f"values_{cr_number}.csv"
    file_path = os.path.join(folder_path, file_name)
    
    if os.path.exists(file_path):
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Ensure correct column names
        if df.columns[:3].tolist() != ['l', 'm', 'alm']:
            print(f"Skipping {file_name}: Unexpected column names")
            continue
        
        # Convert 'alm' column to complex numbers
        df['alm'] = df['alm'].apply(lambda x: complex(x))
        
        # Split into real and imaginary parts
        df['alm_real'] = df['alm'].apply(lambda x: x.real)
        df['alm_imag'] = df['alm'].apply(lambda x: x.imag)
        
        # Drop the original 'alm' column
        df.drop(columns=['alm'], inplace=True)
        
        # Save the modified file
        output_file_path = os.path.join(output_folder, file_name)
        df.to_csv(output_file_path, index=False)
        print(f"Processed and saved: {output_file_path}")
    else:
        print(f"File not found: {file_name}")