import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import wandb
from tqdm import tqdm
import random
import os
from typing import Dict, List, Tuple
import multiprocessing as mp
from functools import partial
import pickle
import hashlib

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Configuration
config = {
    'batch_size': 128,
    'lr': 0.001,
    'epochs': 100,
    'hidden_dim': 128,
    'dropout': 0.3,
    'checkpoint_dir': "ckpt_baseline",
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': mp.cpu_count() // 2,  # Use half of available CPU cores
    'cache_dir': "feature_cache"  # Directory to store pre-calculated features
}

# Create cache directory if it doesn't exist
os.makedirs(config['cache_dir'], exist_ok=True)

# Define category mapping
CATEGORIES = {
    'Blueball': 0,
    'Box': 1,
    'Pencilcase': 2,
    'Pinkball': 3,
    'StuffedAnimal': 4,
    'Tennis': 5,
    'Waterbottle': 6,
}

# Path to the folder containing the dataset files
folder_path = "data_20250414"

# Stats trackers
total_count = 0
kept_count = 0
valid_file_count = 0
skipped_due_to_missing_waypoints = 0

# Define waypoints for robot arm movement
WAYPOINTS = [
    (30, -30), (30, 30), (15, -30), (15, 30),
    (0, -30), (0, 30), (-15, -30), (-15, 30),
    (-30, -30), (-30, 30), (-30, -30), (30, -30),
    (30, 30), (-30, 30)
]

# Step 1: Load and label dataset
def load_and_label_file(file_path, file_name):
    global total_count
    category = next((key for key in CATEGORIES if key in file_name), None)
    if category is None:
        return pd.DataFrame()

    data = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 10:
                try:
                    timestamp = parts[0]
                    sec = float(parts[1])/1000000  # convert microsec to sec for stability
                    x = float(parts[2])
                    y = float(parts[3])
                    x_target = float(parts[4])
                    y_target = float(parts[5])
                    pwm1 = int(parts[6])
                    pwm2 = int(parts[7])
                    pwm3 = int(parts[8])
                    pwm4 = int(parts[9])
                    total_count += 1

                    data.append([
                        timestamp, sec, x, y, x_target, y_target,
                        pwm1, pwm2, pwm3, pwm4, category, CATEGORIES[category]
                    ])
                except ValueError:
                    continue

    return pd.DataFrame(data, columns=[
        "timestamp", "seconds", "x", "y", "x_target", "y_target",
        "pwm1", "pwm2", "pwm3", "pwm4", "category", "label"
    ])

# Step 2: Assign sequential waypoint numbers
def assign_sequential_waypoints(df, tol=1.0):
    df = df.reset_index(drop=True)
    wp_index = 0
    assigned_wp = []

    for i in range(len(df)):
        x_t, y_t = df.loc[i, "x_target"], df.loc[i, "y_target"]
        current_expected = WAYPOINTS[wp_index]

        if np.isclose(x_t, current_expected[0], atol=tol) and np.isclose(y_t, current_expected[1], atol=tol):
            assigned_wp.append(wp_index)
        else:
            if wp_index + 1 < len(WAYPOINTS):
                next_expected = WAYPOINTS[wp_index + 1]
                if np.isclose(x_t, next_expected[0], atol=tol) and np.isclose(y_t, next_expected[1], atol=tol):
                    wp_index += 1
                    assigned_wp.append(wp_index)
                else:
                    assigned_wp.append(wp_index)
            else:
                assigned_wp.append(wp_index)

    df["waypoint_number"] = assigned_wp
    return df

# Step 3: Filter out rows where y <= 0
def filter_by_y(df):
    global kept_count
    filtered = df[df["y"] > 0].reset_index(drop=True)
    kept_count += len(filtered)
    return filtered

def process_file(file_path, file_name):
    global valid_file_count

    # Step 1: Load and label
    df = load_and_label_file(file_path, file_name)
    if df.empty:
        return pd.DataFrame()

    # Step 2: Assign waypoint numbers
    df = assign_sequential_waypoints(df)

    # Step 3: Filter out rows where y ≤ 0
    df = filter_by_y(df)

    # Count as valid if any data was kept
    if not df.empty:
        valid_file_count += 1

    return df

def build_datasets(data_dir: str):
    data_dir = Path(data_dir)
    file_paths = list(data_dir.glob("*.txt"))
    random.seed(42)

    # 1. Group files by object class
    class_to_files = {}
    for class_name in CATEGORIES:
        class_to_files[class_name] = []
    
    for file_path in file_paths:
        for class_name in CATEGORIES:
            if class_name in file_path.name:
                class_to_files[class_name].append(file_path)
                break

    # 2. Stratified split (each class in train/val/test)
    train_files, val_files, test_files = [], [], []
    for class_name, files in class_to_files.items():
        random.shuffle(files)
        n = len(files)
        train_split = int(0.65 * n)
        val_split = int(0.85 * n)
        train_files += files[:train_split]
        val_files += files[train_split:val_split]
        test_files += files[val_split:]

    print("🔍 Per-class file counts:")
    for cls in CATEGORIES:
        print(f"  {cls:<15} → {len(class_to_files[cls])} total files")

    print("\n✅ Final split file counts:")
    print(f"Train: {len(train_files)}")
    print(f"Val:   {len(val_files)}")
    print(f"Test:  {len(test_files)}")

    # 3. Process each split
    def process_file_list(file_list):
        dfs = []
        for fp in file_list:
            df = process_file(fp, fp.name)
            if not df.empty:
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    print("Processing train files")
    train_df = process_file_list(train_files)
    print("Processing validation files")
    val_df = process_file_list(val_files)
    print("Processing test files")
    test_df = process_file_list(test_files)

    return train_df, val_df, test_df

# Extract statistical features from each file
def extract_statistical_features(df):
    if df.empty:
        return None
    
    # Group by waypoint to calculate statistics for each waypoint
    waypoint_groups = df.groupby('waypoint_number')
    
    # Initialize feature dictionary
    features = {}
    
    # Define the expected number of waypoints (from WAYPOINTS list)
    expected_waypoints = len(WAYPOINTS)
    
    # For each expected waypoint, calculate statistics
    for wp in range(expected_waypoints):
        # Check if this waypoint exists in the data
        if wp in waypoint_groups.groups:
            group = waypoint_groups.get_group(wp)
            
            # Basic statistics for position and force sensors
            for col in ['x', 'y', 'pwm1', 'pwm2', 'pwm3', 'pwm4']:
                if col in group.columns:
                    # Handle mean and std calculations safely
                    mean_val = group[col].mean()
                    std_val = group[col].std() if len(group) > 1 else 0.0
                    min_val = group[col].min()
                    max_val = group[col].max()
                    
                    features[f'{col}_wp{wp}_mean'] = mean_val if not pd.isna(mean_val) else 0.0
                    features[f'{col}_wp{wp}_std'] = std_val if not pd.isna(std_val) else 0.0
                    features[f'{col}_wp{wp}_min'] = min_val if not pd.isna(min_val) else 0.0
                    features[f'{col}_wp{wp}_max'] = max_val if not pd.isna(max_val) else 0.0
                    features[f'{col}_wp{wp}_range'] = (max_val - min_val) if not (pd.isna(max_val) or pd.isna(min_val)) else 0.0
            
            # Calculate distance between actual and target positions
            if 'x' in group.columns and 'y' in group.columns and 'x_target' in group.columns and 'y_target' in group.columns:
                distances = np.sqrt((group['x'] - group['x_target'])**2 + (group['y'] - group['y_target'])**2)
                dist_mean = distances.mean()
                dist_std = distances.std() if len(distances) > 1 else 0.0
                
                features[f'distance_wp{wp}_mean'] = dist_mean if not pd.isna(dist_mean) else 0.0
                features[f'distance_wp{wp}_std'] = dist_std if not pd.isna(dist_std) else 0.0
        else:
            # If waypoint doesn't exist, fill with zeros
            for col in ['x', 'y', 'pwm1', 'pwm2', 'pwm3', 'pwm4']:
                features[f'{col}_wp{wp}_mean'] = 0.0
                features[f'{col}_wp{wp}_std'] = 0.0
                features[f'{col}_wp{wp}_min'] = 0.0
                features[f'{col}_wp{wp}_max'] = 0.0
                features[f'{col}_wp{wp}_range'] = 0.0
            
            features[f'distance_wp{wp}_mean'] = 0.0
            features[f'distance_wp{wp}_std'] = 0.0
    
    # Calculate overall statistics across all waypoints
    for col in ['x', 'y', 'pwm1', 'pwm2', 'pwm3', 'pwm4']:
        if col in df.columns:
            # Handle overall statistics safely
            mean_val = df[col].mean()
            std_val = df[col].std() if len(df) > 1 else 0.0
            min_val = df[col].min()
            max_val = df[col].max()
            
            features[f'{col}_overall_mean'] = mean_val if not pd.isna(mean_val) else 0.0
            features[f'{col}_overall_std'] = std_val if not pd.isna(std_val) else 0.0
            features[f'{col}_overall_min'] = min_val if not pd.isna(min_val) else 0.0
            features[f'{col}_overall_max'] = max_val if not pd.isna(max_val) else 0.0
            features[f'{col}_overall_range'] = (max_val - min_val) if not (pd.isna(max_val) or pd.isna(min_val)) else 0.0
    
    # Calculate time-based features safely
    if 'seconds' in df.columns:
        total_time = df['seconds'].max() - df['seconds'].min()
        features['total_time'] = total_time if not pd.isna(total_time) else 0.0
        features['time_per_waypoint'] = total_time / expected_waypoints if expected_waypoints > 0 and not pd.isna(total_time) else 0.0
    
    # Add label
    features['label'] = df['label'].iloc[0]
    features['category'] = df['category'].iloc[0]
    
    return features

# Function to process a single window
def process_window(args):
    window_data, feature_keys = args
    features = extract_statistical_features(window_data)
    if features is not None:
        # Get the label for this window (use the majority label in the window)
        label = window_data['label'].mode().iloc[0]
        
        # Extract features in a consistent order
        feature_values = [features[k] for k in feature_keys]
        
        return feature_values, label
    return None

# Function to generate a cache key for a dataset
def generate_cache_key(df, seq_len):
    # Create a hash of the dataframe content and sequence length
    df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()
    return f"{df_hash}_{seq_len}"

# Create a dataset class for the statistical features
class StatisticalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 2500, use_cache: bool = True):
        self.seq_len = seq_len
        self.df = df.reset_index(drop=True)
        
        # Select feature columns (same as in WindowedDataset)
        self.features = self.df[[
            "x", "y", "x_target", "y_target", "pwm1", "pwm2", "pwm3", "pwm4", "waypoint_number"
        ]].values.astype(np.float32)
        
        # Label per row (same as in WindowedDataset)
        self.labels = self.df["label"].values.astype(np.int64)
        
        # Calculate the number of samples
        self.num_samples = len(self.df) - self.seq_len + 1
        
        # Generate cache key
        cache_key = generate_cache_key(self.df, seq_len)
        cache_file = os.path.join(config['cache_dir'], f"statistical_features_{cache_key}.pkl")
        
        # Check if cached features exist
        if use_cache and os.path.exists(cache_file):
            print(f"Loading pre-calculated features from {cache_file}...")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                self.statistical_features = cached_data['features']
                self.statistical_labels = cached_data['labels']
                self.feature_keys = cached_data['feature_keys']
        else:
            # Get the first window to determine feature keys
            first_window = self.df.iloc[0:self.seq_len]
            first_features = extract_statistical_features(first_window)
            if first_features is None:
                raise ValueError("Failed to extract features from the first window")
            
            # Get the feature keys in a consistent order
            self.feature_keys = sorted([k for k in first_features.keys() if k not in ['label', 'category']])
            
            # Pre-compute statistical features for each window using parallel processing
            print(f"Computing statistical features for {self.num_samples} windows using {config['num_workers']} workers...")
            
            # Prepare arguments for parallel processing
            process_args = []
            for i in range(self.num_samples):
                window_data = self.df.iloc[i:i+self.seq_len]
                process_args.append((window_data, self.feature_keys))
            
            # Process windows in parallel
            with mp.Pool(processes=config['num_workers']) as pool:
                results = list(tqdm(pool.imap(process_window, process_args), total=len(process_args)))
            
            # Filter out None results and separate features and labels
            valid_results = [r for r in results if r is not None]
            if not valid_results:
                raise ValueError("No valid features were extracted from any window")
            
            self.statistical_features = [r[0] for r in valid_results]
            self.statistical_labels = [r[1] for r in valid_results]
            
            # Convert to tensors
            self.statistical_features = torch.tensor(self.statistical_features, dtype=torch.float32)
            self.statistical_labels = torch.tensor(self.statistical_labels, dtype=torch.long)
            
            # Cache the results
            if use_cache:
                print(f"Saving pre-calculated features to {cache_file}...")
                with open(cache_file, 'wb') as f:
                    pickle.dump({
                        'features': self.statistical_features,
                        'labels': self.statistical_labels,
                        'feature_keys': self.feature_keys
                    }, f)
        
        # Print feature dimension
        print(f"Feature dimension: {self.statistical_features.shape[1]}")
    
    def __len__(self):
        return len(self.statistical_labels)
    
    def __getitem__(self, idx):
        return self.statistical_features[idx], self.statistical_labels[idx]

# Define a simple MLP classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super(MLPClassifier, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        logits = self.mlp(x)
        probs = self.softmax(logits)
        return {"out": probs}

# Utility class for tracking metrics
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Calculate accuracy
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# Training function with debugging
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = data
        
        # Debug: Check for NaN or Inf in input data
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"WARNING: NaN or Inf detected in input data at batch {i}")
            print(f"NaN count: {torch.isnan(x).sum().item()}, Inf count: {torch.isinf(x).sum().item()}")
            # Replace NaN and Inf with zeros
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        x, y = x.to(device), y.to(device)
        
        # Debug: Print input statistics
        if i == 0:
            print(f"Input stats - Min: {x.min().item():.4f}, Max: {x.max().item():.4f}, Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}")
        
        outputs = model(x)
        
        # Debug: Check for NaN or Inf in model outputs
        if torch.isnan(outputs['out']).any() or torch.isinf(outputs['out']).any():
            print(f"WARNING: NaN or Inf detected in model outputs at batch {i}")
            print(f"NaN count: {torch.isnan(outputs['out']).sum().item()}, Inf count: {torch.isinf(outputs['out']).sum().item()}")
            # Replace NaN and Inf with small values
            outputs['out'] = torch.nan_to_num(outputs['out'], nan=1e-6, posinf=1.0, neginf=1e-6)
        
        loss = criterion(outputs['out'], y)
        
        # Debug: Check for NaN or Inf in loss
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"WARNING: NaN or Inf detected in loss at batch {i}")
            print(f"Loss value: {loss.item()}")
            # Skip this batch
            continue
        
        loss.backward()
        
        # Debug: Check for NaN or Inf in gradients
        for name, param in model.named_parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                print(f"WARNING: NaN or Inf detected in gradients for {name} at batch {i}")
                print(f"NaN count: {torch.isnan(param.grad).sum().item()}, Inf count: {torch.isinf(param.grad).sum().item()}")
                # Clip gradients
                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        acc = accuracy(outputs['out'], y)[0].item()
        loss_m.update(loss.item())
        acc_m.update(acc)

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(loss_m.avg)),
            acc="{:.04f}%".format(float(acc_m.avg)),
            lr="{:.06f}".format(float(optimizer.param_groups[0]['lr']))
        )
        batch_bar.update()

        del x, y, outputs, loss
        torch.cuda.empty_cache()

    batch_bar.close()
    return loss_m.avg, acc_m.avg

# Validation function
@torch.no_grad()
def validate_model(model, val_loader, criterion, class_names, device):
    model.eval()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    batch_bar = tqdm(total=len(val_loader), dynamic_ncols=True, position=0, leave=False, desc='Val')

    all_preds = []
    all_targets = []

    for i, data in enumerate(val_loader):
        x, y = data
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs['out'], y)

        acc = accuracy(outputs['out'], y)[0].item()

        _, predicted = torch.max(outputs['out'], 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(y.cpu().numpy())

        loss_m.update(loss.item())
        acc_m.update(acc)

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(loss_m.avg)),
            acc="{:.04f}%".format(float(acc_m.avg))
        )
        batch_bar.update()

        del x, y, outputs, loss
        torch.cuda.empty_cache()

    batch_bar.close()

    if class_names:
        print("\nPer-class Validation Accuracy:")
        per_class_acc = {}
        for i, class_name in enumerate(class_names):
            class_mask = (np.array(all_targets) == i)
            if np.sum(class_mask) > 0:
                class_correct = np.sum((np.array(all_preds)[class_mask] == i))
                class_total = np.sum(class_mask)
                acc_percent = 100 * class_correct / class_total
                print(f"  {class_name}: {acc_percent:.4f}% ({class_correct}/{class_total})")
                per_class_acc[f"val_acc_{class_name}"] = acc_percent

    return loss_m.avg, acc_m.avg

# Test function
@torch.no_grad()
def test_model(model, test_loader, criterion, class_names, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []

    class_correct = {class_name: 0 for class_name in class_names}
    class_total = {class_name: 0 for class_name in class_names}

    for data in test_loader:
        inputs, targets = data
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        outputs_for_loss = outputs['out'] if isinstance(outputs, dict) and 'out' in outputs else outputs
        loss = criterion(outputs_for_loss, targets)
        test_loss += loss.item() * inputs.size(0)

        probs = torch.nn.functional.softmax(outputs_for_loss, dim=1)
        _, predicted = torch.max(outputs_for_loss, 1)

        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        for i in range(targets.size(0)):
            label = targets[i].item()
            pred = predicted[i].item()
            class_name = class_names[label]
            class_total[class_name] += 1
            if pred == label:
                class_correct[class_name] += 1

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    test_acc = correct / total

    class_accuracy = {
        name: class_correct[name]/class_total[name] if class_total[name] > 0 else 0
        for name in class_names
    }

    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({correct}/{total})")
    print("\nPer-Class Accuracy:")
    for class_name in class_names:
        print(f" {class_name}: {class_accuracy[class_name]:.4f} ({class_correct[class_name]}/{class_total[class_name]})")

    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'class_accuracy': class_accuracy,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }

# Function to save model
def save_model(model, optimizer, scheduler, metrics, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }, path)

# Function to check data quality
def check_data_quality(dataset, name="dataset"):
    """Check for potential issues in the dataset."""
    print(f"\nChecking {name} quality...")
    
    # Check for NaN or Inf in features
    features = dataset.statistical_features
    if torch.isnan(features).any() or torch.isinf(features).any():
        print(f"WARNING: NaN or Inf detected in {name} features")
        print(f"NaN count: {torch.isnan(features).sum().item()}, Inf count: {torch.isinf(features).sum().item()}")
        
        # Print statistics for debugging
        print(f"Feature stats - Min: {features.min().item():.4f}, Max: {features.max().item():.4f}, Mean: {features.mean().item():.4f}, Std: {features.std().item():.4f}")
        
        # Replace NaN and Inf with zeros
        features = torch.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        dataset.statistical_features = features
    
    # Check for extreme values
    if features.max().item() > 1e6 or features.min().item() < -1e6:
        print(f"WARNING: Extreme values detected in {name} features")
        print(f"Min: {features.min().item():.4f}, Max: {features.max().item():.4f}")
        
        # Normalize features to a reasonable range
        features = torch.clamp(features, min=-1e6, max=1e6)
        dataset.statistical_features = features
    
    # Check for class imbalance
    labels = dataset.statistical_labels
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"Class distribution in {name}:")
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        print(f"  Class {label}: {count} samples ({count/len(labels)*100:.2f}%)")
    
    return dataset

# Main function with debugging
def main():
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Build datasets
    train_df, val_df, test_df = build_datasets(folder_path)
    
    # Create datasets
    seq_len = 2500  # Same as in train_0414.py
    train_dataset = StatisticalDataset(train_df, seq_len=seq_len)
    val_dataset = StatisticalDataset(val_df, seq_len=seq_len)
    test_dataset = StatisticalDataset(test_df, seq_len=seq_len)
    
    # Check data quality
    train_dataset = check_data_quality(train_dataset, "train")
    val_dataset = check_data_quality(val_dataset, "validation")
    test_dataset = check_data_quality(test_dataset, "test")
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model
    input_dim = train_dataset.statistical_features.shape[1]
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_classes=len(CATEGORIES),
        dropout=config['dropout']
    ).to(config['device'])
    
    # Debug: Print model architecture
    from torchinfo import summary
    summary(model, input_size=(config['batch_size'], input_dim))
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=3)
    
    # Initialize wandb
    wandb.login()
    run = wandb.init(
        name="baseline-statistical",
        project="object_classification",
        config=config
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0
    class_names = list(CATEGORIES.keys())
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, config['device'])
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        
        val_loss, val_acc = validate_model(model, val_loader, criterion, class_names, config['device'])
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")
        
        scheduler.step()
        curr_lr = optimizer.param_groups[0]['lr']
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, best_model_path)
            wandb.save(best_model_path)
            print(f"Saved best model with validation loss: {best_val_loss:.4f} and accuracy: {best_val_acc:.2f}%")
        
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'learning_rate': curr_lr
        }, step=epoch)
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model and evaluate on test set
    best_model_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    results = test_model(model, test_loader, criterion, class_names, config['device'])
    
    # Plot confusion matrix
    preds = results['predictions']
    targets = results['targets']
    
    cm = confusion_matrix(targets, preds)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_perc, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title='Confusion Matrix (Percentage)'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            perc = cm_perc[i, j]
            ax.text(j, i, f"{count}\n({perc:.1f}%)",
                    ha="center", va="center",
                    color="white" if perc > 50 else "black")
    
    plt.tight_layout()
    plt.savefig('baseline_confusion_matrix_percentage.png')
    
    # Finish wandb run
    run.finish()

if __name__ == "__main__":
    main() 