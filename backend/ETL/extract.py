import os
import numpy as np
import torchvision.transforms as transforms
from utils.preprocess import EdgeInputHandler, FingernailAnemiaDataset

def is_edge_input(input_path):
    return os.path.isfile(input_path) and input_path.endswith(('.png', '.jpg', '.jpeg'))

def extract_data(dataset_type=None, dataset_dir=None, batch_size=8, tag="[ETL]"):
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

    if dataset_type == "edge-input":
        print(f"{tag} Processing edge input: {dataset_dir}")
        handler = EdgeInputHandler(
            image_path=dataset_dir,
            transform=edge_transform,
            tag=tag,
            save_crops=True,
            debug_dir=os.path.join(dataset_dir if os.path.isdir(dataset_dir) else ".", "debug_crops")
                    )
        dataloader = handler.get_dataloader(batch_size=batch_size)
        for pid, crop, label, hb in dataloader:
            print(f"{pid}: {crop.shape}")
        return dataloader
    
    elif dataset_type == "fingernail-anemia":
        print(f"{tag} Processing dataset: {dataset_dir}")
        handler = FingernailAnemiaDataset(data_dir=dataset_dir, transform=transform, tag=tag)
        train_dataset, test_dataset = handler.get_datasets()
        train_loader, test_loader = handler.get_dataloaders(batch_size=batch_size)
        return [train_dataset, test_dataset], [train_loader, test_loader]