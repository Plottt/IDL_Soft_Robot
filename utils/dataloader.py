import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import re

CATEGORIES = {
    'Blueball': 0,
    'Box': 1,
    'Pencilcase': 2,
    'Pinkball': 3,
    'StuffedAnimal': 4,
    'Tennis': 5,
    'Waterbottle': 6,
}

class ContactDataset(Dataset):
    def __init__(self, data_dir: str, labels: Dict[str, int] = None):
        """
        Args:
            data_dir (str): Directory containing the .txt files
            labels (Dict[str, int]): Dictionary mapping filenames to class labels
        """
        self.data_dir = Path(data_dir)
        self.file_paths = list(self.data_dir.glob("*.txt"))
        self.labels = labels or {}
        # self.label_names = {k: 0 for k in labels.keys()}
        # print(self.labels)
        # print(self.label_names)
        
    def _parse_file(self, file_path: Path) -> pd.DataFrame:
        """Parse a single data file"""
        # Read the file, skip the first line which contains "Waypoint: 0"
        df = pd.read_csv(file_path, header=None, skiprows=1)
        # Define column names based on the data format
        columns = [
            'timestamp_pc',
            'timestamp_micro',
            'x', 'y',
            'angle_1', 'angle_2',
            'contact_1_left', 'contact_1_right',
            'contact_2_left', 'contact_2_right'
        ]
        df = pd.DataFrame(df.values, columns=columns)
        return df
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        file_path = self.file_paths[idx]
        df = self._parse_file(file_path)
        # Create feature vector: Extract relevant features

        features = np.array([
            df['contact_1_left'].mean(),
            df['contact_1_right'].mean(),
            df['contact_2_left'].mean(),
            df['contact_2_right'].mean(),
            df['x'].max() - df['x'].min(),
            df['y'].max() - df['y'].min(),
            df['angle_1'].std(),
            df['angle_2'].std()
        ])

        # Get label from filename (you'll need to implement your labeling logic)
        category = re.sub(r"\d+", "", file_path.stem)
        label = self.labels.get(category, -1)
        
        return torch.FloatTensor(features), label

# dataset = ContactDataset(data_dir="data/IDL_Project", labels=CATEGORIES)
# print(len(dataset))
# print(dataset[0])