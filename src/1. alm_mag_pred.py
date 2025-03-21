import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

dataset_path = "/kaggle/input/alm-magnitudes-spherical-harmonics-solar-physics/mag_alm_values/"
output_folder = "/kaggle/working/predicted"
os.makedirs(output_folder, exist_ok=True)

def find_available_cr_maps(folder):
    """Find all available Carrington Rotation (CR) map files."""
    files = sorted(f for f in os.listdir(folder) if f.startswith("mag_values_") and f.endswith(".csv"))
    cr_numbers = [int(f.split("_")[-1].split(".")[0]) for f in files]
    return cr_numbers

def train_and_predict_intermediate_cr(folder):
    """Train model using alternating CR maps and predict missing ones."""
    available_crs = find_available_cr_maps(folder)
    
    if len(available_crs) < 3:
        print("Not enough data to train the model.")
        return
    
    model = LinearRegression()
    
    for i in range(len(available_crs) - 2):  # Loop over CR pairs
        cr_n = available_crs[i]  # First CR map
        cr_n2 = available_crs[i+2]  # Second CR map
        cr_n1 = available_crs[i+1]  # Target CR map
        
        file_n = os.path.join(folder, f"mag_values_{cr_n}.csv")
        file_n2 = os.path.join(folder, f"mag_values_{cr_n2}.csv")
        file_n1 = os.path.join(folder, f"mag_values_{cr_n1}.csv")
        
        if not os.path.exists(file_n) or not os.path.exists(file_n2):
            continue  # Skip if files are missing
        
        df_n = pd.read_csv(file_n)
        df_n2 = pd.read_csv(file_n2)
        
        # Merge based on l, m to align data
        df = df_n.merge(df_n2, on=['l', 'm'], suffixes=(f'_{cr_n}', f'_{cr_n2}'))
        
        # Features (X) and Target (y)
        X = df[[f'|alm|_{cr_n}', f'|alm|_{cr_n2}']].values
        y = (df[f'|alm|_{cr_n}'] + df[f'|alm|_{cr_n2}']) / 2  # Midpoint estimation
        
        # Train Model
        model.fit(X, y)
        
        # Predict missing CR_n1 values
        df[f'|alm|_pred'] = model.predict(X).round(6)  # Round to 6 decimal places
        
        # Load actual data for CR_n1 if it exists
        if os.path.exists(file_n1):
            df_n1 = pd.read_csv(file_n1)
            df = df.merge(df_n1, on=['l', 'm'], how='left')
        else:
            df['|alm|'] = np.nan  # Create an empty actual column if data is missing
        
        # Save predictions inside Kaggle output folder
        output_file = os.path.join(output_folder, f"predicted_mag_values_{cr_n1}.csv")
        df[['l', 'm', '|alm|', '|alm|_pred']].to_csv(output_file, index=False, float_format="%.6f")
        
        print(f"Predicted and saved: {output_file}")

# Run the script with your dataset
train_and_predict_intermediate_cr(dataset_path)