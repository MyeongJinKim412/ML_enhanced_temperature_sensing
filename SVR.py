import tkinter as tk
from tkinter.filedialog import askopenfilenames, asksaveasfilename
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import re
import os
import random
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, Normalizer
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducible results
random.seed(42)
np.random.seed(42)

# Font settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ---------- File Selection Function ----------
def select_files(title):
    root = tk.Tk()
    root.withdraw()
    files = askopenfilenames(title=title, filetypes=[("CSV files", "*.csv")])
    root.destroy()  # Prevent memory leak
    return sorted(list(files))

# ---------- Data Extraction Functions ----------
def read_xy(file):
    try:
        df = pd.read_csv(file, header=0, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(file, header=0, encoding='cp949')
        except Exception as e:
            print(f"File reading error: {file} - {e}")
            return np.array([]), np.array([])
    
    if df.shape[1] < 4:
        print(f"File format error: {file} - insufficient columns")
        return np.array([]), np.array([])
    
    x = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    y = pd.to_numeric(df.iloc[:, 3], errors='coerce')
    
    # Remove NaN values
    valid_indices = ~(np.isnan(x) | np.isnan(y))
    return x[valid_indices].values, y[valid_indices].values

def parse_info(file_name):
    m = re.search(r'(Set\d+).*?([0-9]+(?:\.[0-9]+)?)\s*[°º]?\s*[C℃]', file_name)
    return (m.group(1), float(m.group(2))) if m else ("Unknown", np.nan)

def extract_peak_ev(x, y):
    if len(x) == 0 or len(y) == 0:
        return np.nan
    idx = np.argmax(y)
    if x[idx] <= 0:  # Prevent division by zero
        return np.nan
    return 1240 / x[idx]

def extract_ex_ratio(x, y):
    if len(x) == 0 or len(y) == 0:
        return np.nan
    main_mask = (x >= 370) & (x <= 380)
    shoulder_mask = (x >= 340) & (x <= 350)
    if not any(main_mask) or not any(shoulder_mask):
        return np.nan
    
    main_max = np.max(y[main_mask])
    shoulder_max = np.max(y[shoulder_mask])
    
    if main_max == 0:  # Prevent division by zero
        return np.nan
    return shoulder_max / main_max

def extract_ratio_400_shoulder(em_x, em_y, ex_x, ex_y):
    if len(ex_x) == 0 or len(ex_y) == 0 or len(em_y) == 0:
        return np.nan
    mask = (340 <= ex_x) & (ex_x <= 350)
    if not any(mask):
        return np.nan
    
    shoulder_max = np.max(ex_y[mask])
    em_max = np.max(em_y)
    
    if em_max == 0:  # Prevent division by zero
        return np.nan
    return shoulder_max / em_max

# ---------- Standardized Correlation Calculation Function ----------
def calculate_standardized_correlations(X_train, y_train, feature_names=None):
    """Calculate and display standardized correlation coefficients"""
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
            try:
                std_corr, p_value = pearsonr(X_std[:, i], y_std)
                standardized_correlations.append((col, std_corr, p_value))
                print(f"{col}: r_std = {std_corr:.4f}, p-value = {p_value:.4f}")
            except Exception as e:
                print(f"Correlation calculation error for {col}: {e}")
                standardized_correlations.append((col, np.nan, np.nan))
    else:
        for i, col in enumerate(feature_names):
            try:
                std_corr, p_value = pearsonr(X_std.iloc[:, i], y_std)
                standardized_correlations.append((col, std_corr, p_value))
                print(f"{col}: r_std = {std_corr:.4f}, p-value = {p_value:.4f}")
            except Exception as e:
                print(f"Correlation calculation error for {col}: {e}")
                standardized_correlations.append((col, np.nan, np.nan))
    
    print("-" * 60)
    return standardized_correlations

# ---------- SVR Hyperparameter Tuning Function ----------
def tune_svr_hyperparameters(X_train, y_train):
    """SVR hyperparameter tuning"""
    print("\n[Starting SVR Hyperparameter Tuning...]")
    print("This process may take some time...")
    
    # Hyperparameter settings for each kernel (removed linear)
    param_grids = {
        'rbf': {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'gamma': ['scale', 0.001, 0.01, 0.1, 1]
        },
        'poly': {
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'gamma': ['scale', 0.001, 0.01, 0.1],
            'degree': [2, 3, 4]
        }
    }
    
    best_models = {}
    best_scores = {}
    
    for kernel in ['rbf', 'poly']:
        print(f"\nTuning {kernel.upper()} kernel...")
        
        try:
            svr = SVR(kernel=kernel)
            grid_search = GridSearchCV(
                svr, 
                param_grids[kernel], 
                cv=5, 
                scoring='r2',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            
            best_models[kernel] = grid_search.best_estimator_
            best_scores[kernel] = grid_search.best_score_
            
            print(f"{kernel.upper()} optimal parameters: {grid_search.best_params_}")
            print(f"{kernel.upper()} CV R2 score: {grid_search.best_score_:.4f}")
            
        except Exception as e:
            print(f"{kernel.upper()} kernel tuning failed: {e}")
            # Replace with default model
            default_model = SVR(kernel=kernel)
            default_model.fit(X_train, y_train)
            best_models[kernel] = default_model
            best_scores[kernel] = 0.0
    
    if not best_models:
        raise ValueError("Tuning failed for all kernels.")
    
    # Select best performing model
    best_kernel = max(best_scores, key=best_scores.get)
    best_model = best_models[best_kernel]
    
    print(f"\nBest performing kernel: {best_kernel.upper()}")
    print(f"Best CV R2 score: {best_scores[best_kernel]:.4f}")
    
    return best_models, best_scores, best_kernel, best_model

# ---------- Model Performance Evaluation Function ----------
def evaluate_models(models, X_train, y_train, X_val, y_val, phase_name):
    """Evaluate and compare performance of multiple models"""
    print(f"\n[{phase_name} Performance Comparison]")
    print("-" * 70)
    
    results = {}
    
    for kernel, model in models.items():
        try:
            # Training performance
            y_train_pred = model.predict(X_train)
            train_r2 = r2_score(y_train, y_train_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            
            # Validation performance
            y_val_pred = model.predict(X_val)
            val_r2 = r2_score(y_val, y_val_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            
            results[kernel] = {
                'train_r2': train_r2, 'train_mae': train_mae, 'train_rmse': train_rmse,
                'val_r2': val_r2, 'val_mae': val_mae, 'val_rmse': val_rmse,
                'predictions': y_val_pred
            }
            
            print(f"{kernel.upper():>8} | Train R2: {train_r2:.4f} | Val R2: {val_r2:.4f} | Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}")
            
        except Exception as e:
            print(f"{kernel.upper()} performance evaluation error: {e}")
            results[kernel] = {
                'train_r2': np.nan, 'train_mae': np.nan, 'train_rmse': np.nan,
                'val_r2': np.nan, 'val_mae': np.nan, 'val_rmse': np.nan,
                'predictions': np.full(len(y_val), np.nan)
            }
    
    print("-" * 70)
    return results

# ---------- Hyperparameter Information Save Function ----------
def save_hyperparameter_info(best_models, best_scores, best_kernel):
    """Save hyperparameter information for each kernel to CSV"""
    
    print("\n[Collecting Hyperparameter Information...]")
    
    # Collect hyperparameter information
    hyperparams_data = []
    
    for kernel, model in best_models.items():
        try:
            params = model.get_params()
            
            # Extract key parameters for each kernel
            row = {
                'Kernel': kernel,
                'CV_R2_Score': best_scores.get(kernel, np.nan),
                'C': params.get('C', 'N/A'),
                'Epsilon': params.get('epsilon', 'N/A'),
                'Gamma': params.get('gamma', 'N/A'),
                'Degree': params.get('degree', 'N/A'),
                'Is_Best_Model': kernel == best_kernel
            }
            
            hyperparams_data.append(row)
            
        except Exception as e:
            print(f"Hyperparameter collection error for {kernel}: {e}")
    
    # Create DataFrame
    hyperparams_df = pd.DataFrame(hyperparams_data)
    
    return hyperparams_df

# ---------- Scaling Information Display Function ----------
def print_scaling_info(scaler, feature_names, scaling_method):
    print(f"\n{scaling_method} Information:")
    
    try:
        if scaling_method == "Standard Scaling (Z-score)":
            for i, feature in enumerate(feature_names):
                mean = scaler.mean_[i]
                scale = scaler.scale_[i]
                print(f"  {feature}: Mean = {mean:.4f}, Std = {scale:.4f}")
        elif scaling_method == "Min-Max Scaling":
            for i, feature in enumerate(feature_names):
                min_val = scaler.data_min_[i]
                max_val = scaler.data_max_[i]
                print(f"  {feature}: Min = {min_val:.4f}, Max = {max_val:.4f}")
        elif scaling_method == "Robust Scaling":
            for i, feature in enumerate(feature_names):
                center = scaler.center_[i]
                scale = scaler.scale_[i]
                print(f"  {feature}: Median = {center:.4f}, IQR = {scale:.4f}")
    except Exception as e:
        print(f"Scaling information display error: {e}")

# ---------- Main Execution ----------

try:
    # Training + Validation data processing
    print("Select training and validation data files...")
    train_val_paths = select_files("Select training and validation data files")
    
    if not train_val_paths:
        print("No files selected.")
        exit()

    print(f"{len(train_val_paths)} files selected.")

    records = []
    processed_count = 0
    
    for file in train_val_paths:
        set_name, temp = parse_info(file)
        if "EmScan" not in file or set_name == "Unknown":
            continue

        related = lambda tag: [f for f in train_val_paths if tag in f and set_name in f]
        f_ex = next((f for f in related("ExScan") if abs(parse_info(f)[1] - temp) <= 0.2), None)
        f_em1 = next((f for f in related("Em(1)") if abs(parse_info(f)[1] - temp) <= 0.2), None)
        f_em2 = next((f for f in related("Em(2)") if abs(parse_info(f)[1] - temp) <= 0.2), None)
        
        if not all([f_ex, f_em1, f_em2]):
            print(f"{set_name} - Missing related files")
            continue

        x_em, y_em = read_xy(file)
        x_ex, y_ex = read_xy(f_ex)
        x_em1, y_em1 = read_xy(f_em1)
        x_em2, y_em2 = read_xy(f_em2)

        if any(len(arr) == 0 for arr in [x_em, y_em, x_ex, y_ex, x_em1, y_em1, x_em2, y_em2]):
            print(f"{set_name} - Data reading failed")
            continue

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
        
        # Check for NaN values
        feature_values = list(row.values())[2:]  # Exclude Set and Temperature
        if not any(np.isnan(feature_values)):
            records.append(row)
            processed_count += 1
        else:
            print(f"{set_name} - Feature calculation failed (contains NaN values)")

    print(f"Processed datasets: {processed_count}")

    # Create DataFrame and split
    df = pd.DataFrame(records)
    if df.empty:
        print("No valid data available for model training.")
        exit()

    print(f"Final dataset: {len(df)} samples")

    X = df.drop(["Temperature", "Set"], axis=1)
    y = df["Temperature"]

    # Train/validation split (80:20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training data: {len(X_train)}, Validation data: {len(X_val)}")

    # Apply SVR-optimized scaling (Standard Scaling)
    original_feature_names = X_train.columns.tolist()

    print(f"\n[Applying Standard Scaling (Z-score) for SVR...]")
    print("SVR is a distance-based algorithm, so feature scaling is essential.")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print(f"Standard Scaling applied.")
    print_scaling_info(scaler, original_feature_names, "Standard Scaling (Z-score)")

    X_train_final = X_train_scaled
    X_val_final = X_val_scaled

    # SVR hyperparameter tuning and model training
    best_models, best_scores, best_kernel, best_model = tune_svr_hyperparameters(X_train_final, y_train)

    # Evaluate all kernel models performance
    train_val_results = evaluate_models(best_models, X_train_final, y_train, X_val_final, y_val, "Train/Validation")

    print(f"\nValidation performance summary by kernel:")
    for kernel in ['rbf', 'poly']:
        if kernel in train_val_results:
            results = train_val_results[kernel]
            print(f"{kernel.upper()} kernel: R2 = {results['val_r2']:.4f}, MAE = {results['val_mae']:.4f}, RMSE = {results['val_rmse']:.4f}")

    # Test data processing
    print("\nSelect test data files...")
    test_paths = select_files("Select test data files")
    
    if not test_paths:
        print("No test files selected.")
        exit()

    test_records = []
    test_processed_count = 0
    
    for file in test_paths:
        set_name, temp = parse_info(file)
        if "EmScan" not in file or set_name == "Unknown":
            continue

        related = lambda tag: [f for f in test_paths if tag in f and set_name in f]
        f_ex = next((f for f in related("ExScan") if abs(parse_info(f)[1] - temp) <= 0.2), None)
        f_em1 = next((f for f in related("Em(1)") if abs(parse_info(f)[1] - temp) <= 0.2), None)
        f_em2 = next((f for f in related("Em(2)") if abs(parse_info(f)[1] - temp) <= 0.2), None)
        
        if not all([f_ex, f_em1, f_em2]):
            print(f"{set_name} - Missing related files")
            continue

        x_em, y_em = read_xy(file)
        x_ex, y_ex = read_xy(f_ex)
        x_em1, y_em1 = read_xy(f_em1)
        x_em2, y_em2 = read_xy(f_em2)

        if any(len(arr) == 0 for arr in [x_em, y_em, x_ex, y_ex, x_em1, y_em1, x_em2, y_em2]):
            print(f"{set_name} - Data reading failed")
            continue

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
        
        # Check for NaN values
        feature_values = list(row.values())[1:]  # Exclude Temperature
        if not any(np.isnan(feature_values)):
            test_records.append(row)
            test_processed_count += 1
        else:
            print(f"{set_name} - Feature calculation failed (contains NaN values)")

    if not test_records:
        print("No valid test data available.")
        exit()

    print(f"Processed test datasets: {test_processed_count}")

    # Test DataFrame and evaluation
    df_test = pd.DataFrame(test_records)
    X_test = df_test.drop(["Temperature"], axis=1)
    y_test = df_test["Temperature"]

    # Apply same scaling to test data
    X_test_final = scaler.transform(X_test)
    print(f"\nStandard Scaling applied to test data")

    # Evaluate test performance with all kernels
    print(f"\n[Test Performance Comparison]")
    print("-" * 70)

    test_results = {}
    for kernel, model in best_models.items():
        try:
            y_test_pred = model.predict(X_test_final)
            test_r2 = r2_score(y_test, y_test_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            test_results[kernel] = {
                'r2': test_r2, 'mae': test_mae, 'rmse': test_rmse,
                'predictions': y_test_pred
            }
            
            print(f"{kernel.upper():>8} | R2: {test_r2:.4f} | MAE: {test_mae:.4f} | RMSE: {test_rmse:.4f}")
        except Exception as e:
            print(f"{kernel.upper()} test prediction error: {e}")
            test_results[kernel] = {
                'r2': np.nan, 'mae': np.nan, 'rmse': np.nan,
                'predictions': np.full(len(y_test), np.nan)
            }

    print("-" * 70)

    print(f"\nTest performance summary by kernel:")
    for kernel in ['rbf', 'poly']:
        if kernel in test_results:
            results = test_results[kernel]
            print(f"{kernel.upper()} kernel: R2 = {results['r2']:.4f}, MAE = {results['mae']:.4f}, RMSE = {results['rmse']:.4f}")

    # Save CSV files for each kernel results
    for kernel in ['rbf', 'poly']:
        if kernel in test_results:
            # Create results DataFrame for each kernel
            df_kernel_results = df_test.copy()
            df_kernel_results[f"Predicted_Temperature_{kernel}"] = test_results[kernel]['predictions']
            df_kernel_results[f"Residual_{kernel}"] = y_test - test_results[kernel]['predictions']
            df_kernel_results[f"Abs_Residual_{kernel}"] = abs(y_test - test_results[kernel]['predictions'])
            
            # Add performance metrics
            df_kernel_results[f"R2_Score_{kernel}"] = test_results[kernel]['r2']
            df_kernel_results[f"MAE_{kernel}"] = test_results[kernel]['mae']
            df_kernel_results[f"RMSE_{kernel}"] = test_results[kernel]['rmse']
            df_kernel_results["Kernel"] = kernel
            
            # Include kernel name in filename
            kernel_filename = f"svr_test_results_{kernel}_scaled.csv"
            
            # File save dialog
            root = tk.Tk()
            root.withdraw()
            save_path = asksaveasfilename(
                title=f"Save SVR {kernel.upper()} Kernel Test Results",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile=kernel_filename
            )
            root.destroy()

            if save_path and not os.path.isdir(save_path):
                # Save test results
                df_kernel_results.to_csv(save_path, index=False)
                print(f"SVR {kernel.upper()} kernel test results saved: {save_path}")
            else:
                print(f"{kernel.upper()} kernel results save skipped")


except Exception as e:
    print(f"\nError occurred during program execution: {e}")
    print("Please check error details and verify data files.")
    import traceback
    traceback.print_exc()