import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class GeneExpressionData(Dataset):
    def __init__(self, data_path, num_genes):
        try:
            # Load data with explicit checks
            data = pd.read_csv(data_path)  # Assuming no header row
            
            # Verify shape
            if data.shape[1] != num_genes + 1:  # genes + label
                raise ValueError(
                    f"Data has {data.shape[1]-1} features (expected {num_genes} genes + 1 label)"
                )
            
            features = data.iloc[:, :num_genes].values.astype(np.float32)
            labels = data.iloc[:, num_genes].values.astype(np.float32)

            valid_mask = (labels >= 0) & (labels <= 1)

            self.features = features[valid_mask]
            self.labels = labels[valid_mask]

            self.features = np.nan_to_num(self.features, nan=0.0, posinf=1e6, neginf=-1e6)
                
            # # Separate features and labels
            # self.features = data.iloc[:, :num_genes].values.astype(np.float32)
            # self.labels = data.iloc[:, num_genes].values.astype(np.float32)
            # self.labels = np.clip(self.labels, 0, 1)
            
            print(f"Successfully loaded data with {len(self.features)} samples, {self.features.shape[1]} genes")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return (
            torch.FloatTensor(self.features[index]),
            torch.FloatTensor([self.labels[index]])
        )