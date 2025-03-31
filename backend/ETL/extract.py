import os
import numpy as np
import torchvision.transforms as transforms
from utils.preprocess import DatasetHandler, EdgeInputHandler

def is_edge_input(input_path):
    return os.path.isfile(input_path) and input_path.endswith(('.png', '.jpg', '.jpeg'))

def extract_data(input_path, batch_size=8, tag="[ETL]"):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=np.random.rand()),
    transforms.RandomVerticalFlip(p=np.random.rand()),
    transforms.RandomRotation(degrees=np.random.randint(0, 360)),
    transforms.RandomAffine(degrees=np.random.randint(0, 360)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    edge_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

    if is_edge_input(input_path):
        print(f"{tag} Processing edge input: {input_path}")
        handler = EdgeInputHandler(image_path=input_path, transform=edge_transform, tag=tag)
        dataloader = handler.get_dataloader(batch_size=batch_size)
        return dataloader
    else:
        print(f"{tag} Processing dataset: {input_path}")
        handler = DatasetHandler(data_dir=input_path, transform=transform, tag=tag)
        train_dataset, test_dataset = handler.get_datasets()
        train_loader, test_loader = handler.get_dataloaders(batch_size=batch_size)
        return [train_dataset, test_dataset], [train_loader, test_loader]
