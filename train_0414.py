import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from torchsummary import summary
from torchinfo import summary
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import wandb
import torch.nn.functional as F
import hashlib
from typing import Dict, Tuple
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: ", device)

# Configuration Dictionary
config = {
    'batch_size': 128,
    'lr': 0.001,
    'epochs': 10,
    'input_dim': 9,
    'num_classes': 7,
    'hidden_dim': 512,
    'num_blocks': 3,
    'checkpoint_dir': "ckpt",
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

if not os.path.exists(config['checkpoint_dir']):
    os.makedirs(config['checkpoint_dir'])

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
                    sec = float(parts[1])/1000000 # convert microsec to sec for stability
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

    # 📌 Show how many waypoints existed before filtering
    waypoint_count_before = df["waypoint_number"].nunique()
    print(f"\n📌 {file_name} → {waypoint_count_before} waypoints BEFORE filtering")

    # Step 3: Filter out rows where y ≤ 0
    df = filter_by_y(df)

    # 📌 Show how many remain after filtering
    waypoint_count_after = df["waypoint_number"].nunique()
    print(f"📌 {file_name} → {waypoint_count_after} waypoints AFTER filtering")

    # Count as valid if any data was kept
    if not df.empty:
        valid_file_count += 1

    return df


# print(all_data.head(1350))


import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import numpy as np


from collections import defaultdict, Counter

def build_datasets(data_dir: str):
    data_dir = Path(data_dir)
    file_paths = list(data_dir.glob("*.txt"))
    random.seed(42)

    # 1. Group files by object class
    class_to_files = defaultdict(list)
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


# 📦 Run everything
train_df, val_df, test_df = build_datasets(folder_path)


class WindowedDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = config['input_dim']):
        self.seq_len = seq_len
        self.df = df.reset_index(drop=True)

        # Select feature columns (change as needed)
        self.features = self.df[[
            "x", "y", "x_target", "y_target", "pwm1", "pwm2", "pwm3", "pwm4", "waypoint_number"
        ]].values.astype(np.float32)

        # Label per row (you can change this to majority/last of the window)
        self.labels = self.df["label"].values.astype(np.int64)

    def __len__(self):
        return len(self.df) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]  # (seq_len, input_dim)
        y = self.labels[idx + self.seq_len - 1]
        x_tensor = torch.tensor(x, dtype=torch.float32)  # ✅ explicitly float32
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor


seq_len = 2500  # adjust as needed

train_dataset = WindowedDataset(train_df, seq_len=seq_len)
val_dataset = WindowedDataset(val_df, seq_len=seq_len)
test_dataset = WindowedDataset(test_df, seq_len=seq_len)

batch_size = config["batch_size"]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



import torch
import torch.nn as nn

class CNNBiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5, lstm_hidden_dim=128, lstm_num_layers=1):
        super(CNNBiLSTMClassifier, self).__init__()

        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Classifier
        # hidden_size * 2 (because bidirectional)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Args:
            x: (B, T, F) where
               B = batch size,
               T = sequence length (time),
               F = feature dimension (input_dim).
        """
        # CNN expects (B, F, T)
        x = x.permute(0, 2, 1)  # -> (B, input_dim, T)

        # Extract features with the CNN
        # output shape: (B, 256, T) after the last conv
        x = self.conv_layers(x)

        # LSTM expects (B, T, Features), so permute back
        x = x.permute(0, 2, 1)  # -> (B, T, 256)

        # Get all time-step outputs from BiLSTM
        lstm_out, (h, c) = self.lstm(x)
        # lstm_out.shape = (B, T, 2*lstm_hidden_dim) for bidirectional
        # h.shape = (2 * num_layers, B, lstm_hidden_dim)

        # Option 1: Take the hidden state from the final time step:
        # final_out = lstm_out[:, -1, :]  # shape: (B, 2*lstm_hidden_dim)

        # Option 2: Take the last hidden state from each of the forward and backward LSTM:
        # (often the same result as above in many LSTM implementations)
        final_forward = h[-2, :, :]  # (B, lstm_hidden_dim), last layer forward
        final_backward = h[-1, :, :] # (B, lstm_hidden_dim), last layer backward
        final_out = torch.cat((final_forward, final_backward), dim=1)  # (B, 2*lstm_hidden_dim)

        # Pass final_out through the classifier
        logits = self.classifier(final_out)
        probs = self.softmax(logits)

        return {"feats": final_out, "out": probs}


model = CNNBiLSTMClassifier(
    input_dim=config['input_dim'],
    num_classes=7,
    dropout=0.5,
    lstm_hidden_dim=128,
    lstm_num_layers=1
).to(config['device'])

from torchinfo import summary
summary(model, input_data=torch.zeros(64, 30, config['input_dim']).to(config['device']))



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


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    loss_m = AverageMeter()
    acc_m = AverageMeter()
    batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = data
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs['out'], y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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


def save_model(model, optimizer, scheduler, metrics, epoch, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }, path)


# Define CrossEntropyLoss as the criterion
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

# Initialize optimizer with AdamW
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['lr'],
    weight_decay=1e-4
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    gamma = 0.5,
    step_size=3
)


import wandb

# Intialize wandb
wandb.login(key="4dd2f46439865db4e3547d39c268ff46468b8ef4") # API Key is in your wandb account, under settings (wandb.ai/settings)


run = wandb.init(
    name = "0414-reproduction", ## Wandb creates random run names if you skip this field
    reinit = False, ### Allows reinitalizing runs when you re-run this cell
    #id = "", ### Insert specific run id here if you want to resume a previous run
    # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
    project = "object_classification", ### Project should be created in your wandb account
    config = config ### Wandb Config for your run
)


# Training Loop
best_val_loss = float('inf')
best_val_acc = 0
class_names = list(CATEGORIES.keys())

for epoch in range(config['epochs']):
    print(f"\nEpoch {epoch + 1}/{config['epochs']}")

    train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, config['device'])
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")

    val_loss, val_acc = validate_model(model, val_loader, criterion, class_names, config['device'])
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

    scheduler.step(val_loss)
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

    last_model_path = os.path.join(config['checkpoint_dir'], f'model_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), last_model_path)
    wandb.save(last_model_path)
    print(f"Saved model for epoch {epoch+1}")

    wandb.log({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'learning_rate': curr_lr
    }, step=epoch)

    print(f"End of Epoch {epoch+1}/{config['epochs']}")

print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")


@torch.no_grad()
def test_model(model, test_loader, criterion, class_names, device, checkpoint_dir=None):
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



best_model_path = f"{config['checkpoint_dir']}/best_model.pth"
if os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")


results = test_model(model, test_loader, criterion, class_names, config['device'])


# Count based confusion matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch


# Convert to numpy if needed
class_names = list(CATEGORIES.keys())

# Convert tensors to numpy
preds = results['predictions']
targets = results['targets']
if isinstance(preds, torch.Tensor):
    preds = preds.cpu().numpy()
if isinstance(targets, torch.Tensor):
    targets = targets.cpu().numpy()

# Compute confusion matrix
cm = confusion_matrix(targets, preds)
cm_sum = cm.sum(axis=1, keepdims=True)
cm_perc = cm / cm_sum.astype(float) * 100

# Plot setup
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax)

# Labels, titles and ticks
ax.set(
    xticks=np.arange(len(class_names)),
    yticks=np.arange(len(class_names)),
    xticklabels=class_names,
    yticklabels=class_names,
    ylabel='True label',
    xlabel='Predicted label',
    title='Confusion Matrix'
)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data to show count and percentage
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        perc = cm_perc[i, j]
        ax.text(j, i, f"{count}\n({perc:.1f}%)",
                ha="center", va="center",
                color="white" if count > thresh else "black")

plt.tight_layout()
# plt.show()
plt.savefig('confusion_matrix_count.png')


import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix

# Suppose CATEGORIES is a dict mapping integer labels to class names
# e.g., CATEGORIES = {0: 'cat', 1: 'dog', 2: 'horse'}
class_names = list(CATEGORIES.keys())

# Convert tensors to numpy if needed
preds = results['predictions']
targets = results['targets']
if isinstance(preds, torch.Tensor):
    preds = preds.cpu().numpy()
if isinstance(targets, torch.Tensor):
    targets = targets.cpu().numpy()

# Compute confusion matrix
cm = confusion_matrix(targets, preds)
cm_sum = cm.sum(axis=1, keepdims=True)
# Percent matrix
cm_perc = cm / cm_sum.astype(float) * 100

# Compute approximate binomial standard error in percentage
# p = count/row_sum; se = sqrt(p*(1-p)/row_sum) * 100
p = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum!=0)
se_perc = np.sqrt(p * (1 - p) / np.maximum(cm_sum, 1)) * 100

# Plot setup
fig, ax = plt.subplots(figsize=(8, 6))

# Color the matrix by percentage, not raw counts
im = ax.imshow(cm_perc, interpolation='nearest', cmap='Blues', vmin=0, vmax=100)
ax.figure.colorbar(im, ax=ax)

# Labels, titles, and ticks
ax.set(
    xticks=np.arange(len(class_names)),
    yticks=np.arange(len(class_names)),
    xticklabels=class_names,
    yticklabels=class_names,
    ylabel='True label',
    xlabel='Predicted label',
    title='Confusion Matrix'
)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Threshold to decide text color
threshold = 50  # or any value in [0,100] that works for your data

# Fill each cell with "XX.X% ± YY.Y%"
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        cell_perc = cm_perc[i, j]
        cell_se = se_perc[i, j]
        # Choose text color based on background intensity
        color = "white" if cell_perc > threshold else "black"
        ax.text(
            j, i, f"{cell_perc:.1f}%",
            ha="center", va="center", color=color
        )

plt.tight_layout()
# plt.show()
plt.savefig('confusion_matrix_percentage.png')



run.finish()


