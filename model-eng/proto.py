print("Importing libraries...")

import os
import sys
import re

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold

from datetime import datetime
import json
import csv

data_dir = "/home/capiku/Workspace/cp-anemia-detection/data/cp-anemia/"
anemic_dir = data_dir+"Anemic/"
nonanemic_dir = data_dir+"Non-anemic/"

print("Loading data sheet...")

data_sheet_path = data_dir+"Anemia_Data_Collection_Sheet.csv"
data_sheet = pd.read_csv(data_sheet_path)
# print(data_sheet)

severity_mapping = {"Non-anemic":0, "Mild":1, "Moderate":2, "Severe":3}
data_sheet['Severity'] = data_sheet['Severity'].map(severity_mapping)

# Sample data sheet
data_sheet = data_sheet.sample(3)

# Define data augmentations or transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=np.random.rand()),
    transforms.RandomVerticalFlip(p=np.random.rand()),
    transforms.RandomRotation(degrees=np.random.randint(0, 360)),
    transforms.RandomAffine(degrees=np.random.randint(0, 360)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom dataset class
class CPAnemiCDataset(Dataset):
    def __init__(self, dir, df, transform=None):
        self.dir = dir
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['IMAGE_ID']
        img_folder = row['REMARK']
        img_path = os.path.join(self.dir, img_folder, img_id + ".png")
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        multiclass_label = torch.tensor(row['Severity'])
        hb_level = torch.tensor(row['HB_LEVEL'])

        return img, multiclass_label, hb_level

print("Loading dataset...")

# Load the dataset
image_dataset = CPAnemiCDataset(data_dir, data_sheet, transform=transform)
train_dataset, test_dataset = train_test_split(image_dataset, test_size=0.20, shuffle=True)

print(f"Image Dataset Size (All): {len(image_dataset)}, \
        Train Size: {len(train_dataset)}, \
        Test Size: {len(test_dataset)}")

BATCH_SIZE = 1
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

for i, (img, label, level) in enumerate(test_loader):
	print(img)
	if i == 3:
		break
