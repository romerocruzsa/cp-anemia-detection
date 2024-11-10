import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import train_test_split

# Transformation definition
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=np.random.randint(0, 360)),
    transforms.RandomAffine(degrees=np.random.randint(0, 360)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
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

        binary_label = torch.tensor(row['REMARK_ONEHOT'], dtype=torch.float32)
        multiclass_label = torch.tensor(row['SEVERITY_ONEHOT'], dtype=torch.float32)
        hb_level = torch.tensor(row['HB_LEVEL'], dtype=torch.float32)
        descriptor = row['REMARK']+"/"+row['Severity']
        
        return img, binary_label, multiclass_label, hb_level, descriptor
    
def transform_data(data_dir):
    data_sheet_path = os.path.join(data_dir, "Anemia_Data_Collection_Sheet.csv")
    data_sheet = pd.read_csv(data_sheet_path)
    # Mapping diagnosis to severity
    severity_mapping = {
        "Non-Anemic": [1,0,0,0],
        "Mild": [0,1,0,0],
        "Moderate": [0,0,1,0],
        "Severe": [0,0,0,1],
    }
    remark_mapping = {
        "Non-anemic": 0,
        "Anemic": 1
    }
    data_sheet['SEVERITY_ONEHOT'] = data_sheet['Severity'].map(severity_mapping)
    data_sheet['REMARK_ONEHOT'] = data_sheet['REMARK'].map(remark_mapping)

    return data_sheet

def load_data(data_dir, test_size=0.20):
    data_sheet = transform_data(data_dir)
    image_dataset = CPAnemiCDataset(data_dir, data_sheet, transform=transform)
    train_dataset, test_dataset = train_test_split(image_dataset, test_size=test_size, shuffle=True)
    print(f"Image Dataset Size (All): {len(image_dataset)}, \
        Train Size: {len(train_dataset)}, \
        Test Size: {len(test_dataset)}")

    return train_dataset, test_dataset