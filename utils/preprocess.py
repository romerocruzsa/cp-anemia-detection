import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score
import os
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from skimage import io as skio
import ast
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class EdgeInputHandler(Dataset):
    def __init__(self, image_path, transform=None, tag=None, crop_size=224, save_crops=False, debug_dir="./debug_crops"):
        self.image_path = image_path
        self.transform = transform
        self.tag = tag
        self.crop_size = crop_size
        self.save_crops = save_crops
        self.debug_dir = debug_dir

        if self.save_crops:
            os.makedirs(self.debug_dir, exist_ok=True)

        print(f"[{self.tag}] Loading image...")
        self.image_rgb = self.load_image()
        print(f"[{self.tag}] Cropping center nailbed region...")
        self.anchor_boxes = self.get_center_crop_box(self.image_rgb.shape[:2])

    def __len__(self):
        return 1  # Always return a single center crop

    def __getitem__(self, idx):
        y1, x1, y2, x2 = self.anchor_boxes[0]
        crop = self.image_rgb[y1:y2, x1:x2, :]

        if self.save_crops:
            debug_path = os.path.join(self.debug_dir, "edge_input.png")
            cv2.imwrite(debug_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        if self.transform:
            crop = self.transform(transforms.ToPILImage()(crop))

        label_class = torch.tensor(0, dtype=torch.long)  # dummy
        hb_level = torch.tensor(0.0, dtype=torch.float32)  # dummy

        return f"edge_input_nail", crop, label_class, hb_level

    def get_dataloader(self, batch_size=1, pin_memory=True):
        return DataLoader(self, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

    def load_image(self):
        image_bgr = cv2.imread(self.image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"{self.tag} Failed to load image: {self.image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb

    def get_center_crop_box(self, image_shape):
        h, w = image_shape
        crop = min(self.crop_size, h, w)
        cx, cy = w // 2, h // 2
        half = crop // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = x1 + crop
        y2 = y1 + crop
        return [(y1, x1, y2, x2)]

    def visualize_anchors(self, save_path=None):
        vis_img = self.image_rgb.copy()
        y1, x1, y2, x2 = self.anchor_boxes[0]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, "Nailbed", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            print(f"{self.tag} Saved center crop visualization to: {save_path}")
        else:
            plt.figure(figsize=(6, 6))
            plt.imshow(vis_img)
            plt.title("Center Nailbed Crop")
            plt.axis("off")
            plt.show()
        
class FingernailFeatures(Dataset):
    def __init__(self, base_dir, df, transform=None):
        self.base_dir = base_dir
        self.df = df
        self.transform = transform
        self.white_refs = self.compute_white_ref()
        self.crop_index = []  # (row_index, 'NAIL', box_idx)

        for idx, row in self.df.iterrows():
            for i in range(3):  # Assuming 3 crops per region
                self.crop_index.append((idx, i))

    def compute_white_ref(self):
        white_refs = {}
        for idx, row in self.df.iterrows():
            label = 'Anemic' if row['HB_LEVEL_GperL'] < 120 else 'Non-anemic'
            img_path = os.path.join(self.base_dir, label, f"{row['PATIENT_ID']}.jpg")
            if not os.path.exists(img_path):
                continue
            img = skio.imread(img_path)
            white_patch = img[350:400, 300:350]  # HxW patch
            medians = [np.median(white_patch[:, :, i]) for i in range(3)]
            white_refs[row['PATIENT_ID']] = medians
        return white_refs

    def __len__(self):
        return len(self.crop_index)

    def __getitem__(self, idx):
        row_idx, box_idx = self.crop_index[idx]
        row = self.df.iloc[row_idx]

        label = 'Anemic' if row['HB_LEVEL_GperL'] < 120 else 'Non-anemic'
        img_path = os.path.join(self.base_dir, label, f"{row['PATIENT_ID']}.jpg")
        image = skio.imread(img_path)

        # WHITE_REF normalization
        if row['PATIENT_ID'] in self.white_refs:
            ref = np.array(self.white_refs[row['PATIENT_ID']]).reshape((1, 1, 3))
            image = np.clip((image / (ref + 1e-6)) * 128, 0, 255).astype(np.uint8)

        nail_boxes = ast.literal_eval(row["NAIL_BOUNDING_BOXES"])
        skin_boxes = ast.literal_eval(row["SKIN_BOUNDING_BOXES"])

        # Nail crop
        t_n, l_n, b_n, r_n = nail_boxes[box_idx]
        nail_crop = image[t_n:b_n, l_n:r_n, :]
        if self.transform:
            nail_crop = self.transform(transforms.ToPILImage()(nail_crop))

        # # Skin crop
        # t_s, l_s, b_s, r_s = skin_boxes[box_idx]
        # skin_crop = image[t_s:b_s, l_s:r_s, :]
        # if self.transform:
        #     skin_crop = self.transform(transforms.ToPILImage()(skin_crop))

        # label_class = torch.tensor(row["SeverityClass"], dtype=torch.long)
        label_class = torch.tensor(row["RemarkClass"], dtype=torch.long)
        hb_level = torch.tensor(row["HB_LEVEL_GperDeciL"], dtype=torch.float32)

        return row["PATIENT_ID"], nail_crop, label_class, hb_level

class FingernailAnemiaDataset:
    def __init__(self, data_dir, transform=None, test_split=0.20, sample_size=None, tag=None, n_per_class=50):
        self.data_dir = data_dir
        self.metadata_path = os.path.join(data_dir, "metadata.csv")
        self.transform = transform
        self.test_split = test_split
        self.sample_size = sample_size
        self.tag = tag
        self.n_per_class = n_per_class

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

        self.data_sheet["HB_LEVEL_GperDeciL"] = self.data_sheet["HB_LEVEL_GperL"].apply(lambda x: x / 10)
        self.data_sheet["RemarkInfo"] = self.data_sheet["HB_LEVEL_GperL"].apply(compute_remark)
        self.data_sheet["SeverityInfo"] = self.data_sheet["HB_LEVEL_GperL"].apply(compute_severity_class)

        self.data_sheet["RemarkClass"] = self.data_sheet["RemarkInfo"].apply(lambda x: x[0])
        self.data_sheet["Remark"] = self.data_sheet["RemarkInfo"].apply(lambda x: x[1])
        self.data_sheet["SeverityClass"] = self.data_sheet["SeverityInfo"].apply(lambda x: x[0])
        self.data_sheet["Severity"] = self.data_sheet["SeverityInfo"].apply(lambda x: x[1])

    def get_features(self):
        return FingernailFeatures(self.data_dir, self.data_sheet, self.transform)

    def get_datasets(self):
        if self.data_sheet is None:
            self.load_data_sheet()

        def kde_balance_by_severity(df, target_column="HB_LEVEL_GperL", severity_column="RemarkClass",
                                    n_per_class=50, bandwidth=0.5, seed=42):
            np.random.seed(seed)
            balanced_dfs = []
            for severity in df[severity_column].unique():
                group = df[df[severity_column] == severity]
                hb_vals = group[target_column].values
                if len(group) <= n_per_class:
                    balanced_dfs.append(group)
                else:
                    kde = gaussian_kde(hb_vals, bw_method=bandwidth / np.std(hb_vals))
                    density = kde(hb_vals)
                    probs = 1.0 / (density + 1e-6)
                    probs /= probs.sum()
                    sampled = group.sample(n=n_per_class, weights=probs, replace=False, random_state=seed)
                    balanced_dfs.append(sampled)
            return pd.concat(balanced_dfs).reset_index(drop=True)

        stratify_col = self.data_sheet["RemarkClass"]
        train_df, test_df = train_test_split(self.data_sheet, test_size=self.test_split, stratify=stratify_col, random_state=42)

        train_df_balanced = kde_balance_by_severity(train_df, n_per_class=self.n_per_class)

        self.train_dataset = FingernailFeatures(self.data_dir, train_df_balanced, self.transform)
        self.test_dataset = FingernailFeatures(self.data_dir, test_df.reset_index(drop=True), self.transform)

        print(f"{self.tag} Dataset loaded — Balanced Train: {len(self.train_dataset)}, Test: {len(self.test_dataset)}")
        return self.train_dataset, self.test_dataset

    def get_dataloaders(self, batch_size=8, pin_memory=True):
        if not self.train_dataset or not self.test_dataset:
            self.get_datasets()
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
        return train_loader, test_loader
