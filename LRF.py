import tkinter as tk
from tkinter.filedialog import askopenfilenames, asksaveasfilename
from tkinter import filedialog, messagebox, simpledialog
import pandas as pd
import numpy as np
import re
import os
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, Normalizer
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Font settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)

# ========== Linear Random Forest Approximate Implementation ==========

class LinearDecisionTree:
    """
    Linear Decision Tree approximate implementation
    Mimics M5 algorithm by applying linear models to leaf nodes
    """
    def __init__(self, max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tree = None
        self.linear_models = {}
        
    def fit(self, X, y):
        """Train Linear Decision Tree with training data"""
        np.random.seed(self.random_state)
        
        # Step 1: Build standard Decision Tree
        self.tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        self.tree.fit(X, y)
        
        # Step 2: Apply linear models to each leaf node (M5 post-processing approximation)
        self._fit_linear_models(X, y)
        
    def _fit_linear_models(self, X, y):
        """Apply linear models to each leaf node"""
        # Identify which leaf each sample belongs to
        leaf_indices = self.tree.apply(X)
        unique_leaves = np.unique(leaf_indices)
        
        for leaf_id in unique_leaves:
            # Samples belonging to this leaf
            mask = leaf_indices == leaf_id
            X_leaf = X[mask]
            y_leaf = y[mask]
            
            # Apply linear model
            if len(X_leaf) >= 2:  # Minimum 2 samples required
                try:
                    linear_model = LinearRegression()
                    linear_model.fit(X_leaf, y_leaf)
                    self.linear_models[leaf_id] = linear_model
                except:
                    # Use mean value if linear model fails
                    self.linear_models[leaf_id] = np.mean(y_leaf)
            else:
                # Use mean value if insufficient samples
                self.linear_models[leaf_id] = np.mean(y_leaf)
    
    def predict(self, X):
        """Perform prediction"""
        leaf_indices = self.tree.apply(X)
        predictions = np.zeros(len(X))
        
        for i, leaf_id in enumerate(leaf_indices):
            if leaf_id in self.linear_models:
                model = self.linear_models[leaf_id]
                if hasattr(model, 'predict'):
                    # Predict with linear model
                    predictions[i] = model.predict(X[i:i+1])[0]
                else:
                    # Use mean value
                    predictions[i] = model
            else:
                # Use default tree prediction
                predictions[i] = self.tree.predict(X[i:i+1])[0]
        
        return predictions

class ApproximateLinearRandomForest:
    """
    Approximate implementation of Linear Random Forest from paper
    
    Key ideas from paper:
    1. Random Sample Bipartition: Split samples into training/pruning sets
    2. Random Feature Selection: Random feature selection at each node
    3. Post-pruning with Linear Models: Apply linear models to leaf nodes
    """
    
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=10, 
                 min_samples_leaf=5, max_features='sqrt', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_ = []
        self.feature_importances_ = None
        
    def fit(self, X, y):
        """Train Linear Random Forest"""
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # Set max_features
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_features = min(self.max_features, n_features)
        else:
            max_features = n_features
            
        self.max_features_ = max_features
        
        print(f"Training Linear Random Forest with {self.n_estimators} trees...")
        print(f"   - Max features per tree: {max_features}")
        print(f"   - Sample bipartition: 80% training, 20% pruning per tree (optimized for small datasets)")
        print(f"   - Training samples per tree: ~{int(n_samples * 0.8)}")
        print(f"   - Pruning samples per tree: ~{int(n_samples * 0.2)}")
        
        # Convert y to numpy array if it's a pandas Series
        if hasattr(y, 'values'):
            y = y.values
        
        # Train each tree
        for i in range(self.n_estimators):
            # 1. Random Sample Bipartition (key idea from paper + improvement for small datasets)
            # Changed from 50:50 to 80:20 split for more training data
            indices = np.random.choice(n_samples, size=n_samples, replace=False)
            train_size = int(n_samples * 0.8)  # 80% training
            train_indices = indices[:train_size]
            prune_indices = indices[train_size:]
            
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_prune = X[prune_indices]
            y_prune = y[prune_indices]
            
            # 2. Random Feature Selection
            feature_indices = np.random.choice(n_features, size=max_features, replace=False)
            
            # 3. Create Linear Decision Tree
            tree = LinearDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + i
            )
            
            # Train with selected features
            tree.fit(X_train[:, feature_indices], y_train)
            
            # Save tree and feature indices
            self.estimators_.append({
                'tree': tree,
                'feature_indices': feature_indices,
                'train_score': r2_score(y_train, tree.predict(X_train[:, feature_indices])),
                'prune_score': r2_score(y_prune, tree.predict(X_prune[:, feature_indices])) if len(X_prune) > 0 else 0
            })
            
            if (i + 1) % 20 == 0:
                print(f"   - Trained {i + 1}/{self.n_estimators} trees")
        
        # Calculate feature importance
        self._calculate_feature_importance(X, y)
        
        print(f"Linear Random Forest training completed!")
        
    def _calculate_feature_importance(self, X, y):
        """Calculate feature importance"""
        n_features = X.shape[1]
        importances = np.zeros(n_features)
        
        for estimator in self.estimators_:
            tree = estimator['tree']
            feature_indices = estimator['feature_indices']
            
            # Use basic tree feature importance
            if hasattr(tree.tree, 'feature_importances_'):
                tree_importances = tree.tree.feature_importances_
                for i, feat_idx in enumerate(feature_indices):
                    if i < len(tree_importances):
                        importances[feat_idx] += tree_importances[i]
        
        # Normalize
        self.feature_importances_ = importances / np.sum(importances) if np.sum(importances) > 0 else importances
        
    def predict(self, X):
        """Perform prediction"""
        predictions = np.zeros((X.shape[0], len(self.estimators_)))
        
        for i, estimator in enumerate(self.estimators_):
            tree = estimator['tree']
            feature_indices = estimator['feature_indices']
            predictions[:, i] = tree.predict(X[:, feature_indices])
        
        # Average ensemble (paper's Equation 5)
        return np.mean(predictions, axis=1)
    
    def get_model_info(self):
        """Return model information"""
        train_scores = [est['train_score'] for est in self.estimators_]
        prune_scores = [est['prune_score'] for est in self.estimators_]
        
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'max_features': self.max_features_,
            'avg_train_score': np.mean(train_scores),
            'avg_prune_score': np.mean(prune_scores),
            'model_type': 'Linear Random Forest (Approximate)'
        }

# ========== Existing Functions (File Processing, Feature Extraction, etc.) ==========

def select_files(title):
    root = tk.Tk()
    root.withdraw()
    files = askopenfilenames(title=title, filetypes=[("CSV files", "*.csv")])
    return sorted(list(files))

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
    if len(x) == 0 or len(y) == 0:
        return np.nan
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
    if not any(mask) or len(em_y) == 0:
        return np.nan
    return np.max(ex_y[mask]) / np.max(em_y)

def plot_feature_importance(model, feature_names, save_dir=None):
    """Visualize feature importance"""
    print("\n[Visualizing feature importance...]")
    
    display_names = {
        'Excitation_ratio': 'Feature (a)',
        'Intensity_400nm_div_shoulder': 'Feature (b)', 
        'EmScan_peak_eV': 'Feature (c)',
        'Em1_peak_eV': 'Feature (d)',
        'Em2_peak_eV': 'Feature (e)'
    }
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Linear Random Forest Feature Importance', fontsize=16, fontweight='bold', fontname='Arial')
    bars = plt.bar(range(len(importances)), importances[indices], alpha=0.8)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(importances)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.xlabel('Features', fontsize=12, fontname='Arial')
    plt.ylabel('Importance', fontsize=12, fontname='Arial')
    
    display_labels = [display_names.get(feature_names[i], feature_names[i]) for i in indices]
    plt.xticks(range(len(importances)), display_labels, rotation=45, ha='right', fontname='Arial')
    plt.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(importances[indices]):
        plt.text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom', fontsize=10, fontname='Arial')
    
    for label in plt.gca().get_xticklabels():
        label.set_fontname('Arial')
    for label in plt.gca().get_yticklabels():
        label.set_fontname('Arial')
    
    plt.tight_layout()
    
    if save_dir:
        importance_plot_path = os.path.join(save_dir, 'lrf_feature_importance_plot.png')
        plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved: {importance_plot_path}")
    
    plt.show()
    
    print("\n[Feature Importance Ranking]")
    print("-" * 70)
    print(f"{'Rank':<6}{'Feature Name':<30}{'Display Name':<20}{'Importance':<10}")
    print("-" * 70)
    for i, idx in enumerate(indices):
        orig_name = feature_names[idx]
        disp_name = display_names.get(orig_name, orig_name)
        print(f"{i+1:<6}{orig_name:<30}{disp_name:<20}{importances[idx]:.4f}")
    print("-" * 70)
    
    return importances, indices

def plot_2d_decision_boundary(model, X_train, y_train, feature_names, top_features_idx, save_dir=None):
    """Visualize 2D decision boundary"""
    print("\n[Visualizing 2D decision boundary...]")

    # Human-readable axis name mapping
    display_names = {
        'Excitation_ratio': 'Feature (a)',
        'Intensity_400nm_div_shoulder': 'Feature (b)',
        'EmScan_peak_eV': 'Feature (c)',
        'Em1_peak_eV': 'Feature (d)',
        'Em2_peak_eV': 'Feature (e)'
    }

    # Selected two key features
    feature1_idx, feature2_idx = top_features_idx[0], top_features_idx[1]
    feature1_name = feature_names[feature1_idx]
    feature2_name = feature_names[feature2_idx]

    feature1_display = display_names.get(feature1_name, feature1_name)
    feature2_display = display_names.get(feature2_name, feature2_name)

    # Training data on 2D plane
    X_2d = X_train[:, [feature1_idx, feature2_idx]]

    # Generate grid
    h = 0.02  # Grid resolution
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Complete prediction grid to match total feature count
    n_features = X_train.shape[1]
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    full_grid = np.zeros((grid_points.shape[0], n_features))
    for i in range(n_features):
        if i == feature1_idx:
            full_grid[:, i] = grid_points[:, 0]
        elif i == feature2_idx:
            full_grid[:, i] = grid_points[:, 1]
        else:
            full_grid[:, i] = np.mean(X_train[:, i])

    # Model prediction
    Z = model.predict(full_grid).reshape(xx.shape)

    # Start visualization
    plt.figure(figsize=(12, 8))

    cmap = 'coolwarm'
    vmin, vmax = 25, 70  # Color bar range (modify if needed)

    # Decision boundary shading
    contour = plt.contourf(xx, yy, Z, levels=20, alpha=0.7, cmap=cmap, vmin=vmin, vmax=vmax)

    # Contour lines
    contour_lines = plt.contour(xx, yy, Z, levels=8, colors='white', alpha=0.8, linewidths=1.5)

    # Contour labels
    labels = plt.clabel(contour_lines, inline=True, fontsize=10, fmt='%.0f°C', inline_spacing=5)

    # Make labels bold
    for label in labels:
        label.set_fontweight('bold')

    # Color bar
    cbar = plt.colorbar(contour, label='Predicted Temperature (°C)')
    cbar.ax.yaxis.label.set_fontname('Arial')

    # Training data scatter plot
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap=cmap, vmin=vmin, vmax=vmax,
                s=80, edgecolor='black', linewidth=1.2, alpha=0.9)

    # Axis and title settings
    plt.xlabel(f'{feature1_display} (Standardized)', fontsize=12, fontname='Arial')
    plt.ylabel(f'{feature2_display} (Standardized)', fontsize=12, fontname='Arial')
    plt.title(f'Linear Random Forest 2D Decision Boundary\n'
              f'(Top 2 features: {feature1_display} vs {feature2_display})',
              fontsize=14, fontweight='bold', fontname='Arial')
    plt.grid(True, alpha=0.3)

    # Unify fonts
    for tick in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        tick.set_fontname('Arial')

    plt.tight_layout()

    # Save image
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        boundary_plot_path = os.path.join(save_dir, 'lrf_decision_boundary_2d.png')
        plt.savefig(boundary_plot_path, dpi=300, bbox_inches='tight')
        print(f"2D decision boundary plot saved: {boundary_plot_path}")

    plt.show()

    # Explanation
    print("\nDecision boundary explanation:")
    print(f"- X-axis: {feature1_display} ({feature1_name}) - standardized values")
    print(f"- Y-axis: {feature2_display} ({feature2_name}) - standardized values")
    print(f"- Points: Training data samples (color = actual temperature)")
    print(f"- Background: Linear Random Forest predicted temperature for each region")


def print_scaling_info(scaler, feature_names):
    """Display standardization information"""
    print("\nStandard Scaling (Z-score) Information:")
    for i, feature in enumerate(feature_names):
        mean = scaler.mean_[i]
        scale = scaler.scale_[i]
        print(f"  {feature}: mean = {mean:.4f}, std = {scale:.4f}")


def print_metrics(name, y_true, y_pred):
    """Display model performance metrics"""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{name} R2: {r2:.4f}, MAE: {mae:.4f}")

# ========== Main Execution Code ==========

print("="*60)
print("Linear Random Forest Temperature Prediction Model")
print("="*60)

# 1. Load training and validation data
print("\n[Step 1: Load Training and Validation Data]")
train_val_paths = select_files("Select training and validation data files")

if not train_val_paths:
    print("No files selected. Exiting program.")
    exit()

print(f"{len(train_val_paths)} files selected.")

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

    try:
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
    except Exception as e:
        print(f"Error processing file: {file}")
        print(f"   Error: {str(e)}")
        continue

# Create DataFrame and split
df = pd.DataFrame(records)
if df.empty:
    print("No valid data available for model training.")
    exit()

print(f"\n{len(df)} valid data sets created.")

X = df.drop(["Temperature", "Set"], axis=1)
y = df["Temperature"]

# 2. Train/validation split (80:20)
print("\n[Step 2: Data Split]")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples")

# 3. Apply scaling
original_feature_names = X_train.columns.tolist()

print(f"\n[Step 3: Apply Standard Scaling]")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"Standard Scaling applied.")
print_scaling_info(scaler, original_feature_names)

X_train_final = X_train_scaled
X_val_final = X_val_scaled

# 4. Create and train Linear Random Forest model
print("\n[Step 4: Linear Random Forest Model Training]")
print("Model Configuration (based on the paper):")
print("  - Sample Bipartition: 80% training, 20% pruning per tree (optimized for small datasets)")
print("  - Linear models in leaf nodes (M5-style post-pruning)")
print("  - Random feature selection at each node")
print("  - Number of trees: 100")
print("  - Max depth: 5")
print("  - Min samples split: 10")
print("  - Min samples leaf: 5")

lrf_model = ApproximateLinearRandomForest(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)

lrf_model.fit(X_train_final, y_train)

# 5. Model evaluation
print(f"\n[Step 5: Model Evaluation]")
model_info = lrf_model.get_model_info()
print(f"Average Tree Training Score: {model_info['avg_train_score']:.4f}")
print(f"Average Tree Pruning Score: {model_info['avg_prune_score']:.4f}")

# Internal performance evaluation
print("\n[Training/Validation Results]")
print_metrics("Training", y_train, lrf_model.predict(X_train_final))
print_metrics("Validation", y_val, lrf_model.predict(X_val_final))

# 6. Visualization
print("\n[Step 6: Visualization]")
root = tk.Tk()
root.withdraw()
viz_save_dir = filedialog.askdirectory(title="Select folder to save visualization results")

if viz_save_dir:
    print(f"Visualization save folder: {viz_save_dir}")
    importances, indices = plot_feature_importance(lrf_model, original_feature_names, viz_save_dir)
    plot_2d_decision_boundary(lrf_model, X_train_final, y_train, original_feature_names, indices, viz_save_dir)
    
    # Save feature importance CSV
    importance_df = pd.DataFrame({
        'Feature': original_feature_names,
        'Importance': importances,
        'Rank': range(1, len(importances) + 1)
    }).sort_values('Importance', ascending=False)
    
    importance_csv_path = os.path.join(viz_save_dir, 'lrf_feature_importance.csv')
    importance_df.to_csv(importance_csv_path, index=False, encoding='utf-8-sig')
    print(f"Feature importance CSV saved: {importance_csv_path}")
else:
    print("No visualization save folder selected. Visualizations will only be displayed.")
    importances, indices = plot_feature_importance(lrf_model, original_feature_names)
    plot_2d_decision_boundary(lrf_model, X_train_final, y_train, original_feature_names, indices)

# 7. Test data processing
print("\n[Step 7: Test Data Processing]")
test_paths = select_files("Select test data files")

if not test_paths:
    print("No test files selected.")
else:
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

        try:
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
        except Exception as e:
            print(f"Error processing test file: {file}")
            print(f"   Error: {str(e)}")
            continue

    if not test_records:
        print("No valid test data available.")
    else:
        # Test DataFrame and evaluation
        df_test = pd.DataFrame(test_records)
        X_test = df_test.drop(["Temperature"], axis=1)
        y_test = df_test["Temperature"]

        # Apply same scaling to test data
        X_test_final = scaler.transform(X_test)
        print(f"\nStandard Scaling applied to test data")

        print("\n[Test Results]")
        y_test_pred = lrf_model.predict(X_test_final)
        print_metrics("Test", y_test, y_test_pred)

        # 8. Save results
        print("\n[Step 8: Save Results]")
        df_test_results = df_test.copy()
        df_test_results["Predicted_Temperature"] = y_test_pred
        df_test_results["Residual"] = y_test - y_test_pred
        df_test_results["Abs_Residual"] = abs(y_test - y_test_pred)

        # Add performance metrics
        r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        df_test_results["R2_Score"] = r2
        df_test_results["MAE"] = mae

        # Add Linear Random Forest model information
        model_info = lrf_model.get_model_info()
        df_test_results["Model_Type"] = model_info['model_type']
        df_test_results["N_Estimators"] = model_info['n_estimators']
        df_test_results["Max_Depth"] = model_info['max_depth']
        df_test_results["Max_Features"] = model_info['max_features']
        df_test_results["Avg_Train_Score"] = model_info['avg_train_score']
        df_test_results["Avg_Prune_Score"] = model_info['avg_prune_score']

        # Include Linear Random Forest information in filename
        scaling_suffix = "_standard_scaled_lrf"
        default_filename = f"lrf_test_results{scaling_suffix}.csv"

        save_path = asksaveasfilename(
            title="Save Linear Random Forest test set prediction results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=default_filename
        )

        if save_path and not os.path.isdir(save_path):
            df_test_results.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"Linear Random Forest test set prediction results saved: {save_path}")
            print(f"Saved columns: Temperature, all feature values, Predicted_Temperature, Residual, Abs_Residual, R2_Score, MAE, Model_Type, N_Estimators, Max_Depth, Max_Features, Avg_Train_Score, Avg_Prune_Score")
        else:
            print("Failed to save test results.")

# 9. Final summary
print(f"\n" + "="*60)
print(f"LINEAR RANDOM FOREST MODEL DETAILS")
print(f"="*60)
print(f"Model Configuration (Based on Paper):")
print(f"  - Algorithm: Linear Random Forest (Approximate Implementation)")
print(f"  - Sample Bipartition: 80% training, 20% pruning per tree (optimized for small datasets)")
print(f"  - Linear models in leaf nodes (M5-style post-pruning)")
print(f"  - Random feature selection: √n_features per tree")
print(f"  - Number of trees: {lrf_model.n_estimators}")
print(f"  - Max depth: {lrf_model.max_depth}")
print(f"  - Min samples split: {lrf_model.min_samples_split}")
print(f"  - Min samples leaf: {lrf_model.min_samples_leaf}")

print(f"\nPerformance Evaluation:")
model_info = lrf_model.get_model_info()
print(f"  - Average Tree Training Score: {model_info['avg_train_score']:.4f}")
print(f"  - Average Tree Pruning Score: {model_info['avg_prune_score']:.4f}")
if 'y_test_pred' in locals():
    print(f"  - Test R2: {r2_score(y_test, y_test_pred):.4f}")
    print(f"  - Test MAE: {mean_absolute_error(y_test, y_test_pred):.4f}")

print(f"\nTop 3 Important Features:")
top_3_features = sorted(zip(original_feature_names, lrf_model.feature_importances_), 
                       key=lambda x: x[1], reverse=True)[:3]
for i, (feature, importance) in enumerate(top_3_features, 1):
    print(f"  {i}. {feature}: {importance:.4f}")

print(f"\nKey Differences from Standard Random Forest:")
print(f"  - Linear models in leaf nodes instead of constant values")
print(f"  - Sample bipartition (80:20) for training and pruning")
print(f"  - Better performance on approximately linear relationships")
print(f"  - Enhanced robustness to data noise and small sample sizes")
print(f"  - Smoother decision boundaries due to linear leaf models")

# Save model hyperparameter summary
if 'viz_save_dir' in locals() and viz_save_dir:
    model_info_dict = {
        'Parameter': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 
                     'max_features', 'random_state', 'model_type', 'sample_bipartition',
                     'avg_train_score', 'avg_prune_score'],
        'Value': [lrf_model.n_estimators, lrf_model.max_depth, lrf_model.min_samples_split, 
                 lrf_model.min_samples_leaf, f"sqrt ({model_info['max_features']})", 
                 lrf_model.random_state, model_info['model_type'], "80% train, 20% prune",
                 f"{model_info['avg_train_score']:.4f}", f"{model_info['avg_prune_score']:.4f}"],
        'Description': ['Number of trees', 'Maximum depth', 'Minimum samples to split', 
                       'Minimum samples in leaf', 'Max features per tree', 'Random seed', 
                       'Algorithm type', 'Sample partitioning strategy',
                       'Average training score', 'Average pruning score']
    }
    
    model_info_df = pd.DataFrame(model_info_dict)
    model_info_path = os.path.join(viz_save_dir, 'linear_random_forest_model_info.csv')
    model_info_df.to_csv(model_info_path, index=False, encoding='utf-8-sig')
    print(f"\nModel info saved: {model_info_path}")