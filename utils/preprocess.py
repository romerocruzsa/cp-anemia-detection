import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from PIL import Image
import ast

class CPAnemic():
    def __init__(self, data_dir, transform=None, test_split=0.2, sample_size=None, tag=None):
        self.data_dir = data_dir
        self.sheet_path = os.path.join(data_dir, "CP-Anemic_Data_Collection_Sheet.csv")
        self.transform = transform
        self.test_split = test_split
        self.sample_size = sample_size
        self.tag = tag

        self.data_sheet = None
        self.train_dataset = None
        self.test_dataset = None

    def load_data_sheet(self):
        print(f"{self.tag} Loading data sheet...")
        self.data_sheet = pd.read_csv(self.sheet_path)
        severity_mapping = {"Non-Anemic": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
        # severity_mapping = {"Non-anemic": 0, "Anemic": 1}
        self.data_sheet["SEVERITY_CLASS"] = self.data_sheet["Severity"].map(severity_mapping)
        # self.data_sheet["SEVERITY_CLASS"] = self.data_sheet["REMARK"].map(severity_mapping)
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
                # binaryclass_label = torch.tensor(row['SEVERITY_CLASS'])
                hb_level = torch.tensor(row['HB_LEVEL'])

                # return img_id, img, multiclass_label, hb_level
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

    def get_dataloaders(self, batch_size=8, pin_memory=False):
        if not self.train_dataset or not self.test_dataset:
            self.get_datasets()
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
        return train_loader, test_loader

class EdgeInputHandler():
    def __init__(self, image_path, transform=None, tag=None):
        self.image_path = image_path
        self.transform = transform
        self.tag = tag

    def load_image(self):
        # Segment first
        print(f"{self.tag} Loading Input Image...")
        self.segment_conjunctiva(save_path="/home/sebastian-cruz6/cp-anemia-detection/output")

        image = Image.open(self.image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image.unsqueeze(0)  # Add batch dimension

    def get_dataloader(self, batch_size=1, pin_memory=True):
        image_tensor = self.load_image()
        dataset = TensorDataset(image_tensor)
        print(f"{self.tag} Image Loaded — Shape: {image_tensor.shape}")
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    def segment_conjunctiva(self, show=False, save_path=None):
        print(f"{self.tag} Loading Segmenting Region-of-Interest...")
        image_bgr = cv2.imread(self.image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"{self.tag} Failed to load image: {self.image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 30, 60])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 30, 60])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 29))
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_mask = np.zeros_like(mask_clean)
        largest_bbox = None
        h, w = mask_clean.shape
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if x > h // 2:
                    cv2.drawContours(output_mask, [cnt], -1, 255, -1)
                    if largest_bbox is None or cw * ch > largest_bbox[2] * largest_bbox[3]:
                            largest_bbox = (x, y, cw, ch)

        # Invert mask to keep only unsegmented areas outside conjunctiva
        output_mask = cv2.bitwise_not(output_mask)

        result = image_rgb.copy()
        result[output_mask == 255] = [0, 0, 0]  # black overlay

        # Center crop based on bounding box
        if largest_bbox:
            x, y, cw, ch = largest_bbox
            cx, cy = x + cw // 2, y + ch // 2
            crop_size = max(cw, ch) * 2
            quarter_crop = crop_size // 4
            x1 = max(0, cx - quarter_crop)
            y1 = max(0, cy - quarter_crop)
            x2 = min(w, cx + quarter_crop)
            y2 = min(h, cy + quarter_crop)
            result = result[y1:y2, x1:x2]
        if save_path:
            # Ensure the save_path ends with a valid image extension
            if not any(save_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg']):
                save_path += "/segmented_output.png"
            cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            
    
        if show:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image_rgb)
            plt.title("Original")
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(result)
            plt.title("Segmented Conjunctiva")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return output_mask, result

class EyesDefyAnemia:
    def __init__(self, data_dir, transform=None, test_split=0.2, sample_size=None, tag=None):
        self.data_dir = data_dir
        self.sheet_path = os.path.join(data_dir, "Eyes-Defy_Data_Collection_Sheet.csv")
        self.transform = transform
        self.test_split = test_split
        self.sample_size = sample_size
        self.tag = tag

        self.data_sheet = None
        self.train_dataset = None
        self.test_dataset = None

    def load_data_sheet(self):
        print(f"{self.tag} Loading Eyes-Defy-Anemia metadata sheet...")
        self.data_sheet = pd.read_csv(self.sheet_path)
        if self.sample_size:
            self.data_sheet = self.data_sheet.sample(self.sample_size)

        def compute_multiclass_label(row):
            hb = row['Hb']
            sex = str(row.get("Sex", "")).strip().upper()
            try:
                hb = float(hb)
                if hb < 8.0:
                    return 3  # Severe
                elif hb < 10.0:
                    return 2  # Moderate
                elif hb < 12.0:
                    return 1  # Mild
                else:
                    return 0  # Non-anemic
            except:
                return -1

        self.data_sheet["SEVERITY_CLASS"] = self.data_sheet.apply(compute_multiclass_label, axis=1)

    def get_features(self):
        class EyesDefyDataset(Dataset):
            def __init__(self, base_dir, df, transform=None):
                self.base_dir = base_dir
                self.df = df
                self.transform = transform
                self.valid_indices = self._validate_images()

            def _validate_images(self):
                valid = []
                for i in range(len(self.df)):
                    row = self.df.iloc[i]
                    label = row['Label']+"/palpebral"
                    img_path = os.path.join(self.base_dir, label, row['Palpebral'] if pd.notna(row['Palpebral']) else row['Image'])
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        valid.append(i)
                    except Exception as e:
                        continue
                        # print(f"[Warning] Skipping invalid image at idx {i}: {img_path}")
                return valid

            def __len__(self):
                return len(self.valid_indices)

            def __getitem__(self, idx):
                try:
                    row_idx = self.valid_indices[idx]
                    row = self.df.iloc[row_idx]
                    label = row['Label']+"/palpebral"
                    hb = row['Hb']
                    severity_class = row['SEVERITY_CLASS']
                    img_path = os.path.join(self.base_dir, label, row['Palpebral'] if pd.notna(row['Palpebral']) else row['Image'])

                    image = Image.open(img_path).convert('RGB')

                    if self.transform:
                        image = self.transform(image)

                    multiclass_label = torch.tensor(severity_class, dtype=torch.long)
                    hb_level = torch.tensor(hb, dtype=torch.float32) if pd.notna(hb) else torch.tensor(-1.0)

                    return row['Image'], image, multiclass_label, hb_level

                except Exception as e:
                    print(f"[Warning] Skipping idx {idx} due to unexpected error")

        return EyesDefyDataset(self.data_dir, self.data_sheet, self.transform)

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
class FingernailAnemiaDataset:
    def __init__(self, data_dir, transform=None, test_split=0.2, sample_size=None, tag=None):
        self.data_dir = data_dir
        self.metadata_path = os.path.join(data_dir, "metadata.csv")
        self.transform = transform
        self.test_split = test_split
        self.sample_size = sample_size
        self.tag = tag

        self.data_sheet = None
        self.train_dataset = None
        self.test_dataset = None

    def load_data_sheet(self):
        print(f"{self.tag} Loading Fingernail-Anemia metadata sheet...")
        self.data_sheet = pd.read_csv(self.metadata_path)

        if self.sample_size:
            self.data_sheet = self.data_sheet.sample(self.sample_size)

        def compute_severity_class(hb):
            try:
                hb = float(hb)
                if hb < 80:
                    return 3, "Severe"
                elif hb < 100:
                    return 2, "Moderate"
                elif hb < 120:
                    return 1, "Mild"
                else:
                    return 0, "Non-anemic"
            except:
                return -1, "Unknown"
        
        def compute_remark(hb):
            try:
                hb = float(hb)
                if hb < 120:
                    return 1, "Anemic"
                else:
                    return 0, "Non-anemic"
            except:
                return -1, "Unknown"

        self.data_sheet["RemarkInfo"] = self.data_sheet["HB_LEVEL_GperL"].apply(compute_remark)
        self.data_sheet["SeverityInfo"] = self.data_sheet["HB_LEVEL_GperL"].apply(compute_severity_class)
        
        self.data_sheet["RemarkClass"] = self.data_sheet["RemarkInfo"].apply(lambda x: x[0])
        self.data_sheet["Remark"] = self.data_sheet["RemarkInfo"].apply(lambda x: x[1])
        self.data_sheet["SeverityClass"] = self.data_sheet["SeverityInfo"].apply(lambda x: x[0])
        self.data_sheet["Severity"] = self.data_sheet["SeverityInfo"].apply(lambda x: x[1])

    def get_features(self):
        class FingernailDataset(Dataset):
            def __init__(self, base_dir, df, transform=None):
                self.base_dir = base_dir
                self.df = df
                self.transform = transform

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                row = self.df.iloc[idx]
                remark = row["Remark"]
                patient_id = row["PATIENT_ID"]
                img_path = os.path.join(self.base_dir, remark, f"{patient_id}.jpg")

                image = Image.open(img_path).convert("RGB")
                image_np = torch.tensor(np.array(image))

                nail_boxes = ast.literal_eval(row["NAIL_BOUNDING_BOXES"])
                skin_boxes = ast.literal_eval(row["SKIN_BOUNDING_BOXES"])

                nail_crops = []
                skin_crops = []

                for box in nail_boxes:
                    t, l, b, r = box
                    crop = image.crop((l, t, r, b))
                    if self.transform:
                        crop = self.transform(crop)
                    nail_crops.append(crop)

                for box in skin_boxes:
                    t, l, b, r = box
                    crop = image.crop((l, t, r, b))
                    if self.transform:
                        crop = self.transform(crop)
                    skin_crops.append(crop)

                nail_tensor = torch.stack(nail_crops)  # (3, C, H, W)
                skin_tensor = torch.stack(skin_crops)  # (3, C, H, W)

                label = torch.tensor(row["SeverityClass"], dtype=torch.long)
                hb_level = torch.tensor(row["HB_LEVEL_GperL"], dtype=torch.float32)

                return patient_id, nail_tensor, skin_tensor, label, hb_level

        return FingernailDataset(self.data_dir, self.data_sheet, self.transform)

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
