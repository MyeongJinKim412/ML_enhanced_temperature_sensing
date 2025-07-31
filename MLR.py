import tkinter as tk
from tkinter.filedialog import askopenfilenames, asksaveasfilename
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import re
import os
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, Normalizer
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Set random seed for reproducible results
random.seed(42)
np.random.seed(42)

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ---------- File selection function ----------
def select_files(title):
    root = tk.Tk()
    root.withdraw()
    files = askopenfilenames(title=title, filetypes=[("CSV files", "*.csv")])
    return sorted(list(files))

# ---------- Data extraction functions ----------
def read_xy(file):
    try:
        df = pd.read_csv(file, header=0, encoding='utf-8')
    except:
        df = pd.read_csv(file, header=0, encoding='cp949')
    x = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    y = pd.to_numeric(df.iloc[:, 3], errors='coerce')
    return x.dropna().values, y.dropna().values

def parse_info(file_name):
    m = re.search(r'(Set\d+).*?([0-9]+(?:\.[0-9]+)?)\s*[°º]?\s*[C℃]', file_name)
    return (m.group(1), float(m.group(2))) if m else ("Unknown", np.nan)

def extract_peak_ev(x, y):
    idx = np.argmax(y)
    return 1240 / x[idx]

def extract_ex_ratio(x, y):
    main_mask = (x >= 370) & (x <= 380)
    shoulder_mask = (x >= 340) & (x <= 350)
    if not any(main_mask) or not any(shoulder_mask):
        return np.nan
    return np.max(y[shoulder_mask]) / np.max(y[main_mask])

def extract_ratio_400_shoulder(em_x, em_y, ex_x, ex_y):
    mask = (340 <= ex_x) & (ex_x <= 350)
    if not any(mask):
        return np.nan
    return np.max(ex_y[mask]) / np.max(em_y)

# ---------- Standardized correlation calculation and output function ----------
def calculate_standardized_correlations(X_train, y_train, feature_names=None):
    """Calculate and print standardized correlation coefficients"""
    print("\n[Standardized Correlation Coefficients]")
    print("-" * 60)
    
    if feature_names is None:
        if hasattr(X_train, 'columns'):
            feature_names = X_train.columns
        else:
            feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
    
    # Data standardization
    X_std = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    y_std = (y_train - np.mean(y_train)) / np.std(y_train)
    
    standardized_correlations = []
    
    if isinstance(X_std, np.ndarray):
        for i, col in enumerate(feature_names):
            std_corr, p_value = pearsonr(X_std[:, i], y_std)
            standardized_correlations.append((col, std_corr, p_value))
            print(f"{col}: r_std = {std_corr:.4f}, p-value = {p_value:.4f}")
    else:
        for i, col in enumerate(feature_names):
            std_corr, p_value = pearsonr(X_std.iloc[:, i], y_std)
            standardized_correlations.append((col, std_corr, p_value))
            print(f"{col}: r_std = {std_corr:.4f}, p-value = {p_value:.4f}")
    
    print("-" * 60)
    return standardized_correlations

# ---------- CSV generation function for correlation plot ----------
def save_correlation_plot_data(standardized_correlations):
    """Generate CSV data for correlation plot"""
    
    print("\n[Generating CSV data for correlation plot...]")
    
    # Select save folder
    root = tk.Tk()
    root.withdraw()
    save_dir = filedialog.askdirectory(title="Select folder to save correlation plot data")
    
    if not save_dir:
        print("ERROR: No save folder selected.")
        return
    
    # File suffix (standardized fixed)
    suffix = "_standard"
    
    # Create correlation data DataFrame
    corr_df = pd.DataFrame(standardized_correlations, columns=['Feature', 'Std_Correlation', 'P_value'])
    corr_df['Abs_Correlation'] = abs(corr_df['Std_Correlation'])
    corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
    
    # Generate grid data for plot (uploaded image style)
    n_features = len(standardized_correlations)
    
    # Generate grid data by normalizing correlation values between -1 and 1
    correlation_values = corr_df['Std_Correlation'].values
    
    # Generate grid data for each correlation coefficient
    plot_data = []
    
    for i, (feature, corr_val, p_val, abs_corr) in enumerate(corr_df.values):
        # Generate pattern based on correlation value
        n_points = 1000  # Number of points per grid
        
        # Generate X, Y coordinates (range -1 to 1)
        x_coords = np.random.uniform(-1, 1, n_points)
        y_coords = np.random.uniform(-1, 1, n_points)
        
        # Apply pattern based on correlation coefficient
        if abs(corr_val) > 0.8:  # Strong correlation
            if corr_val > 0:  # Positive correlation
                # Diagonal pattern
                mask = abs(y_coords - x_coords) < 0.3
            else:  # Negative correlation
                # Anti-diagonal pattern
                mask = abs(y_coords + x_coords) < 0.3
        elif abs(corr_val) > 0.5:  # Medium correlation
            if corr_val > 0:
                # Wide diagonal pattern
                mask = abs(y_coords - x_coords) < 0.5
            else:
                # Wide anti-diagonal pattern
                mask = abs(y_coords + x_coords) < 0.5
        else:  # Weak correlation
            # Circular or random pattern
            if abs(corr_val) < 0.1:
                # Completely random
                mask = np.ones(n_points, dtype=bool)
            else:
                # Elliptical pattern
                mask = (x_coords**2 + (y_coords/corr_val)**2) < 1
        
        # Apply mask
        x_filtered = x_coords[mask]
        y_filtered = y_coords[mask]
        
        # Save data
        for x, y in zip(x_filtered, y_filtered):
            plot_data.append({
                'Feature': feature,
                'Feature_Index': i,
                'Correlation': corr_val,
                'P_value': p_val,
                'X': x,
                'Y': y
            })
    
    # Convert to DataFrame
    plot_df = pd.DataFrame(plot_data)
    
    # Save CSV file
    plot_file = os.path.join(save_dir, f'correlation_plot_data{suffix}.csv')
    plot_df.to_csv(plot_file, index=False)
    print(f"SUCCESS: Correlation plot data saved: {plot_file}")
    
    # Save correlation summary information file
    summary_file = os.path.join(save_dir, f'correlation_summary{suffix}.csv')
    corr_df.to_csv(summary_file, index=False)
    print(f"SUCCESS: Correlation summary information saved: {summary_file}")
    
    print(f"\nCorrelation plot file generation completed!")
    print(f"Save location: {save_dir}")
    print(f"Main file: correlation_plot_data{suffix}.csv")
    print(f"Features to analyze: {n_features}")
    
    return save_dir

# ---------- Scaling information output function ----------
def print_scaling_info(scaler, feature_names):
    print(f"\nStandard Scaling (Z-score) information:")
    
    for i, feature in enumerate(feature_names):
        mean = scaler.mean_[i]
        scale = scaler.scale_[i]
        print(f"  {feature}: mean = {mean:.4f}, std = {scale:.4f}")

# ---------- Main execution section ----------

# Training + validation data processing
train_val_paths = select_files("Select training and validation data files")

records = []
for file in train_val_paths:
    set_name, temp = parse_info(file)
    if "EmScan" not in file or set_name == "Unknown":
        continue

    related = lambda tag: [f for f in train_val_paths if tag in f and set_name in f]
    f_ex = next((f for f in related("ExScan") if abs(parse_info(f)[1] - temp) <= 0.2), None)
    f_em1 = next((f for f in related("Em(1)") if abs(parse_info(f)[1] - temp) <= 0.2), None)
    f_em2 = next((f for f in related("Em(2)") if abs(parse_info(f)[1] - temp) <= 0.2), None)
    if not all([f_ex, f_em1, f_em2]):
        continue

    x_em, y_em = read_xy(file)
    x_ex, y_ex = read_xy(f_ex)
    x_em1, y_em1 = read_xy(f_em1)
    x_em2, y_em2 = read_xy(f_em2)

    temps = [temp, parse_info(f_ex)[1], parse_info(f_em1)[1], parse_info(f_em2)[1]]
    T_avg = round(np.mean(temps), 2)

    row = {
        "Set": set_name,
        "Temperature": T_avg,
        "Excitation_ratio": extract_ex_ratio(x_ex, y_ex),
        "EmScan_peak_eV": extract_peak_ev(x_em, y_em),
        "Em1_peak_eV": extract_peak_ev(x_em1, y_em1),
        "Em2_peak_eV": extract_peak_ev(x_em2, y_em2),
        "Intensity_400nm_div_shoulder": extract_ratio_400_shoulder(x_em, y_em, x_ex, y_ex),
    }
    if not any(np.isnan(list(row.values())[2:])):
        records.append(row)

# DataFrame creation and splitting
df = pd.DataFrame(records)
if df.empty:
    print("WARNING: No valid data available for model training.")
    exit()

X = df.drop(["Temperature", "Set"], axis=1)
y = df["Temperature"]

# Training/validation split (80:20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply scaling (Standard Scaling fixed)
original_feature_names = X_train.columns.tolist()

print(f"\n[Applying Standard Scaling (Z-score)...]")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"SUCCESS: Standard Scaling has been applied.")
print_scaling_info(scaler, original_feature_names)

X_train_final = X_train_scaled
X_val_final = X_val_scaled

# Model training
model = LinearRegression()
model.fit(X_train_final, y_train)

# Internal performance evaluation
print("\n[Training/Validation Results]")
def print_metrics(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name} R2: {r2:.4f}, MAE: {mae:.4f}")

print_metrics("Training", y_train, model.predict(X_train_final))
print_metrics("Validation", y_val, model.predict(X_val_final))

# Standardized correlation analysis (calculated with original data)
standardized_correlations = calculate_standardized_correlations(X_train, y_train, original_feature_names)

# Test data processing
test_paths = select_files("Select test data files")

test_records = []
for file in test_paths:
    set_name, temp = parse_info(file)
    if "EmScan" not in file or set_name == "Unknown":
        continue

    related = lambda tag: [f for f in test_paths if tag in f and set_name in f]
    f_ex = next((f for f in related("ExScan") if abs(parse_info(f)[1] - temp) <= 0.2), None)
    f_em1 = next((f for f in related("Em(1)") if abs(parse_info(f)[1] - temp) <= 0.2), None)
    f_em2 = next((f for f in related("Em(2)") if abs(parse_info(f)[1] - temp) <= 0.2), None)
    if not all([f_ex, f_em1, f_em2]):
        continue

    x_em, y_em = read_xy(file)
    x_ex, y_ex = read_xy(f_ex)
    x_em1, y_em1 = read_xy(f_em1)
    x_em2, y_em2 = read_xy(f_em2)

    temps = [temp, parse_info(f_ex)[1], parse_info(f_em1)[1], parse_info(f_em2)[1]]
    T_avg = round(np.mean(temps), 2)
    
    row = {
        "Temperature": T_avg,
        "Excitation_ratio": extract_ex_ratio(x_ex, y_ex),
        "EmScan_peak_eV": extract_peak_ev(x_em, y_em),
        "Em1_peak_eV": extract_peak_ev(x_em1, y_em1),
        "Em2_peak_eV": extract_peak_ev(x_em2, y_em2),
        "Intensity_400nm_div_shoulder": extract_ratio_400_shoulder(x_em, y_em, x_ex, y_ex),
    }
    if not any(np.isnan(list(row.values())[1:])):
        test_records.append(row)

if not test_records:
    print("WARNING: No valid test data available.")
    exit()

# Test DataFrame and evaluation
df_test = pd.DataFrame(test_records)
X_test = df_test.drop(["Temperature"], axis=1)
y_test = df_test["Temperature"]

# Apply same scaling to test data
X_test_final = scaler.transform(X_test)
print(f"\nSUCCESS: Standard Scaling applied to test data")

print("\n[Test Results]")
y_test_pred = model.predict(X_test_final)
print_metrics("Test", y_test, y_test_pred)

# Generate CSV data for correlation plot
save_correlation_plot_data(standardized_correlations)

# Save result CSV file
df_test_results = df_test.copy()
df_test_results["Predicted_Temperature"] = y_test_pred
df_test_results["Residual"] = y_test - y_test_pred
df_test_results["Abs_Residual"] = abs(y_test - y_test_pred)

# Add performance metrics
r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
df_test_results["R2_Score"] = r2
df_test_results["MAE"] = mae

# Reflect scaling method in filename (standardized fixed)
scaling_suffix = "_standard_scaled"
default_filename = f"lr_test_results{scaling_suffix}.csv"

save_path = asksaveasfilename(
    title="Save test set prediction results",
    defaultextension=".csv",
    filetypes=[("CSV files", "*.csv")],
    initialfile=default_filename
)

if save_path and not os.path.isdir(save_path):
    df_test_results.to_csv(save_path, index=False)
    print(f"SUCCESS: Test set prediction results saved: {save_path}")
    print(f"Saved columns: Temperature, all features, Predicted_Temperature, Residual, Abs_Residual, R2_Score, MAE")
else:
    print("ERROR: Failed to save test results.")

# Final summary
print(f"Scaling: Standard Scaling (Z-score)")
print(f"Total {len(standardized_correlations)} features analyzed")
print(f"Test performance: R2 = {r2_score(y_test, y_test_pred):.4f}, MAE = {mean_absolute_error(y_test, y_test_pred):.4f}")