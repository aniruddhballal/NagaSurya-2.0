import os
import pandas as pd
from sklearn.linear_model import LinearRegression

dataset_path = "alm_values_split"
output_folder = "predicted_alm_values"
os.makedirs(output_folder, exist_ok=True)

def find_available_cr_maps(folder):
    """Find all available Carrington Rotation (CR) map files."""
    files = sorted(f for f in os.listdir(folder) if f.startswith("values_") and f.endswith(".csv"))
    cr_numbers = sorted([int(f.split("_")[-1].split(".")[0]) for f in files])  # Extract CR numbers
    return cr_numbers

def train_and_predict_all_crs(folder):
    """Train model bidirectionally to predict ALL missing CR maps."""
    available_crs = find_available_cr_maps(folder)
    
    if len(available_crs) < 3:
        print("Not enough data to train the model.")
        return
    
    model_real = LinearRegression()
    model_imag = LinearRegression()
    
    for i in range(1, len(available_crs) - 1):  # Ensuring bidirectional prediction
        cr_prev = available_crs[i - 1]  # Previous CR map
        cr_next = available_crs[i + 1]  # Next CR map
        cr_target = available_crs[i]    # Target CR map to predict
        
        file_prev = os.path.join(folder, f"values_{cr_prev}.csv")
        file_next = os.path.join(folder, f"values_{cr_next}.csv")
        
        if not os.path.exists(file_prev) or not os.path.exists(file_next):
            continue  # Skip if any required file is missing
        
        df_prev = pd.read_csv(file_prev)
        df_next = pd.read_csv(file_next)
        
        # Merge based on (l, m) to align data
        df = df_prev.merge(df_next, on=['l', 'm'], suffixes=('_prev', '_next'))
        
        # Features (X) and Target (y) for real and imaginary parts
        X_real = df[['alm_real_prev', 'alm_real_next']].values
        y_real = (df['alm_real_prev'] + df['alm_real_next']) / 2  # Midpoint estimation

        X_imag = df[['alm_imag_prev', 'alm_imag_next']].values
        y_imag = (df['alm_imag_prev'] + df['alm_imag_next']) / 2  # Midpoint estimation
        
        # Train Model
        model_real.fit(X_real, y_real)
        model_imag.fit(X_imag, y_imag)
        
        # Predict missing CR_target values
        df['alm_real'] = model_real.predict(X_real).round(6)  # Round to 6 decimal places
        df['alm_imag'] = model_imag.predict(X_imag).round(6)  # Round to 6 decimal places
        
        # Save predictions inside Kaggle output folder
        output_file = os.path.join(output_folder, f"predicted_values_{cr_target}.csv")
        df[['l', 'm', 'alm_real', 'alm_imag']].to_csv(output_file, index=False, float_format="%.6f")
        
        print(f"Predicted and saved: {output_file}")

# Run the script with your dataset
train_and_predict_all_crs(dataset_path)