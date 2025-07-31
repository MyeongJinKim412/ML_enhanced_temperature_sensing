import tkinter as tk
from tkinter.filedialog import askopenfilenames, asksaveasfilename
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import re
import os
import random
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
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
    return sorted(list(files))

# ---------- Data Extraction Functions ----------
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

# ---------- Pyramid MLP Model Creation ----------
class PyramidMLPRegressor(MLPRegressor):
    """Extended sklearn MLPRegressor class that also records validation loss"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.val_loss_curve_ = []
        self.X_val = None
        self.y_val = None
    
    def set_validation_data(self, X_val, y_val):
        """Set validation data"""
        self.X_val = X_val
        self.y_val = y_val
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Override original fit method to record validation loss"""
        
        # Set validation data
        if X_val is not None and y_val is not None:
            self.set_validation_data(X_val, y_val)
        
        # Call original sklearn fit method (maintain performance)
        result = super().fit(X, y)
        
        # Reconstruct validation losses for each epoch after training
        if self.X_val is not None and self.y_val is not None:
            self._reconstruct_validation_losses(X, y)
        
        return result
    
    def _reconstruct_validation_losses(self, X_train, y_train):
        """Reconstruct validation losses for each epoch after training completion"""
        print("Reconstructing validation losses...")
        
        # Backup current trained weights
        final_coefs = [coef.copy() for coef in self.coefs_]
        final_intercepts = [intercept.copy() for intercept in self.intercepts_]
        
        # Create temporary model to calculate epoch-wise validation losses
        temp_model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate_init=self.learning_rate_init,
            max_iter=1,  # 1 epoch at a time
            early_stopping=False,
            random_state=self.random_state,
            shuffle=self.shuffle,
            warm_start=True
        )
        
        self.val_loss_curve_ = []
        
        # Re-train model epoch by epoch while recording validation loss
        for epoch in range(len(self.loss_curve_)):
            temp_model.partial_fit(X_train, y_train)
            
            # Calculate validation loss
            val_pred = temp_model.predict(self.X_val)
            val_loss = mean_squared_error(self.y_val, val_pred)
            self.val_loss_curve_.append(val_loss)
            
            # Print progress (optional)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1:3d}: Validation loss = {val_loss:.6f}")
        
        # Restore original weights
        self.coefs_ = final_coefs
        self.intercepts_ = final_intercepts
        
        print("Validation loss reconstruction completed!")

def create_pyramid_mlp():
    """Create pyramid MLP model"""
    
    model = PyramidMLPRegressor(
        hidden_layer_sizes=(256, 128, 64, 32, 16),  # Pyramid structure
        activation='relu',                          # ReLU activation
        alpha=0.01,                                # L2 regularization
        batch_size=32,                             # Batch size
        learning_rate_init=0.01,                   # Learning rate
        max_iter=2000,                             # Maximum iterations
        early_stopping=True,                       # Early stopping
        validation_fraction=0.1,                   # Validation fraction
        n_iter_no_change=20,                      # Early stopping criterion
        random_state=42,                           # Reproducibility
        shuffle=True                               # Shuffle data
    )
    
    return model

# ---------- PDP/ICE Analysis Function (MLP Version) ----------
def analyze_mlp_pdp_ice(model, X_data, feature_names, save_dir=None):
    """
    PDP and ICE analysis for MLP model
    
    Parameters:
    -----------
    model : sklearn model
        Trained MLP model
    X_data : numpy array
        Scaled input data
    feature_names : list
        Feature name list
    save_dir : str, optional
        Save directory path
    """
    print("\n[MLP PDP/ICE Analysis Started]")
    print("Pyramid MLP model Partial Dependence Plot (PDP) and Individual Conditional Expectation (ICE) analysis")
    print("Detecting nonlinearity and heterogeneous response patterns by individual samples")
    print("-" * 90)
    
    if save_dir is None:
        root = tk.Tk()
        root.withdraw()
        save_dir = filedialog.askdirectory(title="Select folder to save MLP PDP/ICE analysis results")
        if not save_dir:
            print("No save folder selected. Skipping analysis.")
            return
    
    # List to store all PDP/ICE data
    all_pdp_ice_data = []
    
    # PDP/ICE analysis for each feature
    for feature_idx, feature_name in enumerate(feature_names):
        print(f"\nAnalyzing: {feature_name} (Feature {feature_idx + 1}/{len(feature_names)})")
        
        try:
            # Calculate PDP and ICE data
            pd_results = partial_dependence(
                model, X_data, [feature_idx], 
                kind='both',  # Calculate both PDP and ICE
                grid_resolution=50  # Grid resolution
            )
            
            # Extract results
            pdp_values = pd_results['average'][0]  # PDP values
            ice_values = pd_results['individual'][0]  # ICE values (for each sample)
            grid_values = pd_results['grid_values'][0]  # X-axis values
            
            # Quantify nonlinearity
            # Calculate PDP curvature (mean of absolute second derivative)
            if len(pdp_values) >= 3:
                second_derivative = np.diff(pdp_values, n=2)
                curvature = np.mean(np.abs(second_derivative))
            else:
                curvature = 0
            
            # ICE line variance (mean of variance at each grid point)
            ice_variability = np.mean(np.var(ice_values, axis=0))
            
            # Calculate nonlinearity indicators
            pdp_range = np.max(pdp_values) - np.min(pdp_values)
            ice_std = np.std(ice_values, axis=0).mean()
            
            # Calculate nonlinearity indicator
            linearity_score = max(0, 1 - (curvature * 100 + ice_variability / (pdp_range + 1e-10)))
            if linearity_score > 0.8:
                nonlinearity_level = "Weak Nonlinearity"
            elif linearity_score > 0.5:
                nonlinearity_level = "Moderate Nonlinearity"
            else:
                nonlinearity_level = "Strong Nonlinearity"
            
            # Prepare CSV data
            for grid_idx, grid_val in enumerate(grid_values):
                # PDP data
                all_pdp_ice_data.append({
                    'Feature_Name': feature_name,
                    'Feature_Index': feature_idx,
                    'Grid_Value': grid_val,
                    'PDP_Value': pdp_values[grid_idx],
                    'Type': 'PDP',
                    'Sample_Index': -1,  # PDP is average, so -1
                    'ICE_Value': np.nan,
                    'PDP_Curvature': curvature,
                    'ICE_Variability': ice_variability,
                    'Linearity_Score': linearity_score,
                    'Nonlinearity_Level': nonlinearity_level
                })
                
                # ICE data (for each sample)
                for sample_idx in range(ice_values.shape[0]):
                    all_pdp_ice_data.append({
                        'Feature_Name': feature_name,
                        'Feature_Index': feature_idx,
                        'Grid_Value': grid_val,
                        'PDP_Value': pdp_values[grid_idx],
                        'Type': 'ICE',
                        'Sample_Index': sample_idx,
                        'ICE_Value': ice_values[sample_idx, grid_idx],
                        'PDP_Curvature': curvature,
                        'ICE_Variability': ice_variability,
                        'Linearity_Score': linearity_score,
                        'Nonlinearity_Level': nonlinearity_level
                    })
            
            print(f"  {feature_name} analysis completed - {nonlinearity_level} detected")
            
        except Exception as e:
            print(f"  Error analyzing {feature_name}: {str(e)}")
            continue
    
    # Save all PDP/ICE data to CSV
    if all_pdp_ice_data:
        pdp_ice_df = pd.DataFrame(all_pdp_ice_data)
        csv_filename = 'MLP_PDP_ICE_Analysis_Data.csv'
        csv_path = os.path.join(save_dir, csv_filename)
        pdp_ice_df.to_csv(csv_path, index=False)
        print(f"\nAll MLP PDP/ICE data saved: {csv_filename}")
        
        # Generate summary statistics (expanded for MLP)
        summary_stats = []
        for feature_name in feature_names:
            feature_data = pdp_ice_df[pdp_ice_df['Feature_Name'] == feature_name]
            pdp_data = feature_data[feature_data['Type'] == 'PDP']
            ice_data = feature_data[feature_data['Type'] == 'ICE']
            
            if not pdp_data.empty and not ice_data.empty:
                pdp_range = pdp_data['PDP_Value'].max() - pdp_data['PDP_Value'].min()
                ice_variability = ice_data.groupby('Grid_Value')['ICE_Value'].std().mean()
                curvature = pdp_data['PDP_Curvature'].iloc[0]
                linearity_score = pdp_data['Linearity_Score'].iloc[0]
                nonlinearity_level = pdp_data['Nonlinearity_Level'].iloc[0]
                
                # Maximum difference between ICE lines
                ice_spread = []
                for grid_val in pdp_data['Grid_Value'].unique():
                    grid_ice_values = ice_data[ice_data['Grid_Value'] == grid_val]['ICE_Value']
                    if len(grid_ice_values) > 1:
                        ice_spread.append(grid_ice_values.max() - grid_ice_values.min())
                
                max_ice_spread = max(ice_spread) if ice_spread else 0
                
                summary_stats.append({
                    'Feature_Name': feature_name,
                    'PDP_Range_DegC': pdp_range,
                    'PDP_Curvature': curvature,
                    'Avg_ICE_Std_DegC': ice_variability,
                    'Max_ICE_Spread_DegC': max_ice_spread,
                    'Grid_Points': len(pdp_data),
                    'Total_Samples': len(ice_data) // len(pdp_data),
                    'Linearity_Score': linearity_score,
                    'Nonlinearity_Level': nonlinearity_level,
                    'Model_Type': 'Pyramid MLP (256→128→64→32→16)',
                    'Total_Parameters': 'Calculated',
                    'Activation_Function': 'ReLU'
                })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_filename = 'MLP_PDP_ICE_Summary.csv'
        summary_path = os.path.join(save_dir, summary_filename)
        summary_df.to_csv(summary_path, index=False)
        print(f"MLP PDP/ICE summary statistics saved: {summary_filename}")
        
        print(f"\nMLP PDP/ICE analysis completed!")
        print(f"Save location: {save_dir}")
        print(f"Total analyzed data points: {len(all_pdp_ice_data):,}")
        
        # Output nonlinearity analysis results
        print(f"\nMLP model feature-wise nonlinearity analysis summary:")
        print("-" * 60)
        for _, row in summary_df.iterrows():
            print(f"{row['Feature_Name']}:")
            print(f"  PDP Range: {row['PDP_Range_DegC']:.3f}°C")
            print(f"  PDP Curvature: {row['PDP_Curvature']:.4f}")
            print(f"  ICE Avg Std: {row['Avg_ICE_Std_DegC']:.3f}°C")
            print(f"  ICE Max Spread: {row['Max_ICE_Spread_DegC']:.3f}°C")
            print(f"  Nonlinearity Level: {row['Nonlinearity_Level']}")
            print(f"  Linearity Score: {row['Linearity_Score']:.3f}")
        
        return save_dir
    else:
        print("Failed to generate MLP PDP/ICE data.")
        return None

# ---------- Scaling Information Display Function ----------
def print_scaling_info(scaler, feature_names):
    print(f"\nStandard Scaling (Z-score) Information:")
    
    for i, feature in enumerate(feature_names):
        mean = scaler.mean_[i]
        scale = scaler.scale_[i]
        print(f"  {feature}: mean = {mean:.4f}, std = {scale:.4f}")

# ---------- Learning Curve Analysis Function ----------
def plot_learning_curves(model, save_dir=None):
    """Save training and validation MSE Loss data to CSV (no plot display)"""
    
    if not hasattr(model, 'loss_curve_'):
        print("No learning curve data available.")
        return None, None
    
    # Training loss (sklearn built-in)
    train_losses = model.loss_curve_
    
    # Validation loss (recorded by custom method)
    val_losses = model.val_loss_curve_ if hasattr(model, 'val_loss_curve_') else []
    
    print("Method: sklearn performance + validation loss recording")
    
    # Detailed analysis (without plotting)
    min_train_loss_epoch = np.argmin(train_losses) + 1
    min_train_loss_value = min(train_losses)
    
    print(f"\nPyramid MLP Training Analysis:")
    print(f"  Network structure: {model.hidden_layer_sizes}")
    print(f"  Activation function: {model.activation}")
    print(f"  Total epochs: {len(train_losses)}")
    print(f"  Final training loss (MSE): {train_losses[-1]:.6f}")
    print(f"  Minimum training loss (MSE): {min_train_loss_value:.6f} (epoch {min_train_loss_epoch})")
    
    if val_losses:
        min_val_loss_epoch = np.argmin(val_losses) + 1
        min_val_loss_value = min(val_losses)
        print(f"  Final validation loss (MSE): {val_losses[-1]:.6f}")
        print(f"  Minimum validation loss (MSE): {min_val_loss_value:.6f} (epoch {min_val_loss_epoch})")
        
        # Overfitting analysis
        final_train_mse = train_losses[-1]
        final_val_mse = val_losses[-1]
        min_val_train_ratio = min_val_loss_value / train_losses[min_val_loss_epoch - 1]
        
        if final_val_mse > final_train_mse * 1.5:
            print(f"  Warning: Possible overfitting - final validation loss is {(final_val_mse/final_train_mse-1)*100:.1f}% higher than training loss")
        elif min_val_train_ratio < 0.8:
            print(f"  Warning: Possible underfitting - ratio at min validation loss = {min_val_train_ratio:.3f}")
        else:
            print(f"  Appropriate learning - ratio at min validation loss = {min_val_train_ratio:.3f}")
    
    return train_losses, val_losses

# ---------- Learning Curve Data Save Function ----------
def save_learning_curve_data(model, save_dir):
    """Save training/validation MSE loss curve data to CSV"""
    
    if not hasattr(model, 'loss_curve_'):
        print("No learning curve data available.")
        return None
    
    # Prepare data
    train_losses = model.loss_curve_
    val_losses = model.val_loss_curve_ if hasattr(model, 'val_loss_curve_') else []
    epochs = range(1, len(train_losses) + 1)
    
    # Calculate loss change rates
    train_loss_change = [0]  # First epoch has 0 change rate
    for i in range(1, len(train_losses)):
        change = train_losses[i-1] - train_losses[i]  # Decrease from previous epoch
        change_rate = (change / train_losses[i-1]) * 100 if train_losses[i-1] != 0 else 0
        train_loss_change.append(change_rate)
    
    # Create DataFrame
    data = {
        'Epoch': epochs,
        'Training_Loss_MSE': train_losses,
        'Train_Loss_Change_Rate_Percent': train_loss_change
    }
    
    # Add validation loss data (if available)
    if val_losses and len(val_losses) == len(train_losses):
        val_loss_change = [0]
        for i in range(1, len(val_losses)):
            change = val_losses[i-1] - val_losses[i]
            change_rate = (change / val_losses[i-1]) * 100 if val_losses[i-1] != 0 else 0
            val_loss_change.append(change_rate)
        
        data['Validation_Loss_MSE'] = val_losses
        data['Val_Loss_Change_Rate_Percent'] = val_loss_change
        data['Val_Train_Loss_Ratio'] = [v/t if t != 0 else np.inf for v, t in zip(val_losses, train_losses)]
    
    # Add minimum loss epoch information
    min_train_loss_epoch = np.argmin(train_losses) + 1
    data['Is_Min_Training_Loss_Epoch'] = [epoch == min_train_loss_epoch for epoch in epochs]
    
    if val_losses:
        min_val_loss_epoch = np.argmin(val_losses) + 1
        data['Is_Min_Validation_Loss_Epoch'] = [epoch == min_val_loss_epoch for epoch in epochs]
    
    learning_curve_df = pd.DataFrame(data)
    
    # Save CSV file
    if save_dir:
        csv_file = os.path.join(save_dir, 'pyramid_mlp_train_val_learning_data.csv')
        learning_curve_df.to_csv(csv_file, index=False)
        print(f"Training/validation learning curve data saved: {csv_file}")
        
        # Also save summary statistics file
        total_params = sum([model.coefs_[i].size + model.intercepts_[i].size for i in range(len(model.coefs_))])
        summary_stats = {
            'Metric': [
                'Total_Epochs',
                'Min_Training_Loss_Epoch',
                'Min_Training_Loss_MSE',
                'Final_Training_Loss_MSE',
                'Initial_Training_Loss_MSE',
                'Training_Loss_Improvement_Percent',
                'Network_Structure',
                'Total_Parameters',
                'Activation_Function',
                'Learning_Rate',
                'Alpha_Regularization',
                'Batch_Size',
                'Loss_Function',
                'Method'
            ],
            'Value': [
                len(train_losses),
                min_train_loss_epoch,
                min(train_losses),
                train_losses[-1],
                train_losses[0],
                ((train_losses[0] - train_losses[-1]) / train_losses[0]) * 100,
                str(model.hidden_layer_sizes),
                total_params,
                model.activation,
                model.learning_rate_init,
                model.alpha,
                model.batch_size,
                'MSE (Mean Squared Error)',
                'sklearn + validation tracking'
            ]
        }
        
        # Add validation loss related information
        if val_losses:
            min_val_loss_epoch = np.argmin(val_losses) + 1
            summary_stats['Metric'].extend([
                'Min_Validation_Loss_Epoch',
                'Min_Validation_Loss_MSE',
                'Final_Validation_Loss_MSE',
                'Val_Train_Loss_Ratio_at_Min_Val_Epoch'
            ])
            summary_stats['Value'].extend([
                min_val_loss_epoch,
                min(val_losses),
                val_losses[-1],
                val_losses[min_val_loss_epoch - 1] / train_losses[min_val_loss_epoch - 1]
            ])
        
        summary_df = pd.DataFrame(summary_stats)
        summary_file = os.path.join(save_dir, 'pyramid_mlp_training_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Training summary information saved: {summary_file}")
    else:
        print("Training/validation learning curve data prepared in memory (not saved - no folder selected)")
    
    return learning_curve_df

# ---------- Main Execution ----------

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

# Create DataFrame
df = pd.DataFrame(records)
if df.empty:
    print("No valid data available for model training.")
    exit()

X = df.drop(["Temperature", "Set"], axis=1)
y = df["Temperature"]

# Train/validation split (80:20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply scaling
original_feature_names = X_train.columns.tolist()

print(f"\n[Applying Standard Scaling (Z-score)...]")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"Standard Scaling applied.")
print_scaling_info(scaler, original_feature_names)

# Create and train pyramid MLP model
print(f"\n[Training Pyramid MLP Model...]")
print(f"Architecture: (256, 128, 64, 32, 16)")
print(f"Activation function: ReLU")
print(f"Regularization: α = 0.01")
print(f"Batch size: 32")
print(f"Method: sklearn performance + additional validation loss recording")

model = create_pyramid_mlp()

# Training (sklearn fit + validation loss recording)
model.fit(X_train_scaled, y_train, X_val_scaled, y_val)

print(f"\nPyramid MLP model training completed!")
print(f"Training iterations: {model.n_iter_}")
print(f"Final training loss: {model.loss_:.6f}")
if hasattr(model, 'val_loss_curve_') and model.val_loss_curve_:
    print(f"Final validation loss: {model.val_loss_curve_[-1]:.6f}")

# Performance evaluation
print("\n[Training/Validation Results]")
def print_detailed_metrics(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{name}:")
    print(f"  R2: {r2:.4f}")
    print(f"  MAE: {mae:.4f}°C")
    print(f"  RMSE: {rmse:.4f}°C") 
    print(f"  MAPE: {mape:.2f}%")

print_detailed_metrics("Training", y_train, model.predict(X_train_scaled))
print_detailed_metrics("Validation", y_val, model.predict(X_val_scaled))

# MLP PDP/ICE Analysis
print("\n" + "="*90)
print("MLP PDP/ICE analysis starting!")
print("This analysis shows nonlinear patterns learned by the pyramid MLP model.")
print("PDP: Average nonlinear effects")
print("ICE: Individual sample nonlinear effects") 
print("Discover nonlinearity and individual sample heterogeneity!")
print("="*90)

# Perform MLP PDP/ICE analysis with full training data
mlp_pdp_save_dir = analyze_mlp_pdp_ice(
    model=model,
    X_data=X_train_scaled,  # Use scaled training data
    feature_names=original_feature_names
)

# Learning curve analysis and save
print(f"\n[Pyramid MLP Learning Curve Analysis]")

# Select save directory (can use same directory as PDP/ICE)
if mlp_pdp_save_dir:
    save_dir = mlp_pdp_save_dir
    print(f"Using same directory as PDP/ICE analysis: {save_dir}")
else:
    root = tk.Tk()
    root.withdraw()
    save_dir = filedialog.askdirectory(title="Select folder to save learning curve data")

if save_dir:
    # Analyze learning curves (no plot display)
    train_losses, val_losses = plot_learning_curves(model, save_dir)
    
    # Save learning curve data to CSV
    learning_curve_df = save_learning_curve_data(model, save_dir)
    
    print(f"Learning curve files saved: {save_dir}")
    print(f"Generated files:")
    print(f"   - pyramid_mlp_train_val_learning_data.csv (learning curve data)")
    print(f"   - pyramid_mlp_training_summary.csv (training summary)")
else:
    # Analyze without saving
    train_losses, val_losses = plot_learning_curves(model)
    learning_curve_df = save_learning_curve_data(model, None)
    print("Learning curve analysis completed (not saved)")
    print("Learning curve data prepared in memory")

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
    print("No valid test data available.")
    exit()

# Test DataFrame and evaluation
df_test = pd.DataFrame(test_records)
X_test = df_test.drop(["Temperature"], axis=1)
y_test = df_test["Temperature"]

# Test data scaling
X_test_scaled = scaler.transform(X_test)
print(f"\nStandard Scaling applied to test data")

print("\n[Final Test Results]")
y_test_pred = model.predict(X_test_scaled)
print_detailed_metrics("Test", y_test, y_test_pred)

# Save results
df_test_results = df_test.copy()
df_test_results["Predicted_Temperature"] = y_test_pred
df_test_results["Residual"] = y_test - y_test_pred
df_test_results["Abs_Residual"] = abs(y_test - y_test_pred)

# Add performance metrics
r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
df_test_results["R2_Score"] = r2
df_test_results["MAE"] = mae
df_test_results["RMSE"] = rmse

# Save file
default_filename = "pyramid_mlp_results.csv"

save_path = asksaveasfilename(
    title="Save Pyramid MLP test results",
    defaultextension=".csv",
    filetypes=[("CSV files", "*.csv")],
    initialfile=default_filename
)

if save_path and not os.path.isdir(save_path):
    df_test_results.to_csv(save_path, index=False)
    print(f"Pyramid MLP results saved: {save_path}")
    print(f"Saved columns: Temperature, all feature values, Predicted_Temperature, Residual, Abs_Residual, R2_Score, MAE, RMSE")
else:
    print("Failed to save test results.")