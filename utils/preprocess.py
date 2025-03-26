import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image

class DatasetHandler():
    def __init__(self, data_dir, transform=None, test_split=0.2, sample_size=None, tag=None):
        self.data_dir = data_dir
        self.anemic_dir = os.path.join(data_dir, "Anemic")
        self.nonanemic_dir = os.path.join(data_dir, "Non-anemic")
        self.sheet_path = os.path.join(data_dir, "Anemia_Data_Collection_Sheet.csv")
        self.transform = transform
        self.test_split = test_split
        self.sample_size = sample_size
        self.tag = tag

        self.data_sheet = None
        self.train_dataset = None
        self.test_dataset = None

    def load_data_sheet(self):
        print(f"{self.tag} Loading data sheet...")
        self.data_sheet = pd.read_csv(self.sheet_path)[:3]
        severity_mapping = {"Non-anemic": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
        self.data_sheet["SEVERITY_CLASS"] = self.data_sheet["Severity"].map(severity_mapping)

        if self.sample_size:
            self.data_sheet = self.data_sheet.sample(self.sample_size)

    def get_features(self):
        class FeatureDataset(Dataset):
            def __init__(self, base_dir, df, transform=None):
                self.base_dir = base_dir
                self.df = df
                self.transform = transform

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                img_id = row['IMAGE_ID']
                img_folder = row['REMARK']
                img_path = os.path.join(self.base_dir, img_folder, img_id + ".png")
                img = Image.open(img_path).convert('RGB')

                if self.transform:
                    img = self.transform(img)

                multiclass_label = torch.tensor(row['SEVERITY_CLASS'])
                hb_level = torch.tensor(row['HB_LEVEL'])

                return img_id, img, multiclass_label, hb_level

        return FeatureDataset(self.data_dir, self.data_sheet, self.transform)

    def get_datasets(self):
        if self.data_sheet is None:
            self.load_data_sheet()
        dataset = self.get_features()
        train_set, test_set = train_test_split(dataset, test_size=self.test_split, shuffle=True)
        self.train_dataset = train_set
        self.test_dataset = test_set

        print(f"{self.tag} Dataset loaded — Total: {len(dataset)}, Train: {len(train_set)}, Test: {len(test_set)}")
        return train_set, test_set

    def get_dataloaders(self, batch_size=8, pin_memory=True):
        if not self.train_dataset or not self.test_dataset:
            self.get_datasets()
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
        return train_loader, test_loader
    
    # def save_to_json(self, output_path=" "):
    #     if self.data_sheet is None:
    #         self.load_data_sheet()

    #     output_path += "/preprocessed.json"
    #     data_to_save = []

    #     for idx, row in self.data_sheet.iterrows():
    #         row_dict = row.to_dict()

    #         # Get image and labels from dataset
    #         dataset = self.get_features()
    #         image_tensor, severity_class, hb_level = dataset[idx]

    #         # Convert tensors to JSON-serializable formats
    #         row_dict["IMAGE_PATH"] = os.path.join(self.data_dir, row['REMARK'], row['IMAGE_ID'] + ".png")
    #         row_dict["IMAGE_VECTOR"] = image_tensor.tolist()
    #         row_dict["SEVERITY_CLASS"] = float(severity_class.item())
    #         row_dict["HB_LEVEL"] = float(hb_level.item())

    #         data_to_save.append(row_dict)

    #     with open(output_path, "w") as f:
    #         json.dump(data_to_save, f, indent=4)

    #     print(f"{self.tag} Saved {len(data_to_save)} entries to {output_path}")

