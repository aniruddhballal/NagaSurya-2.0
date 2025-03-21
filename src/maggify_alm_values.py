import pandas as pd
import os

def compute_magnitude_for_all():
    input_folder = "alm_values"
    output_folder = "mag_alm_values"
    
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    for carrington_number in range(2096, 2286):  # 2285 inclusive
        input_file = f"{input_folder}/values_{carrington_number}.csv"
        output_file = f"{output_folder}/mag_values_{carrington_number}.csv"
        
        if not os.path.exists(input_file):
            print(f"Skipping {input_file}, file not found.")
            continue
        
        # Load the CSV file
        df = pd.read_csv(input_file)
        
        # Ensure required columns exist
        if 'alm' not in df.columns or 'm' not in df.columns:
            print(f"Skipping {input_file}, required columns missing.")
            continue
        
        # Convert 'alm' column to complex numbers and compute magnitude
        df['|alm|'] = df['alm'].apply(lambda x: abs(complex(x))).astype(float)
        
        # Remove the 'alm' column
        df.drop(columns=['alm'], inplace=True)
        
        # Keep only rows where m >= 0
        df = df[df['m'] >= 0]
        
        # Save the new CSV file
        df.to_csv(output_file, index=False, float_format="%.10f")
        print(f"Saved output to {output_file}")

# Run the function
compute_magnitude_for_all()