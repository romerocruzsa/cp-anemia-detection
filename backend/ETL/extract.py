import os
import numpy as np
import torchvision.transforms as transforms
from utils.preprocess import CPAnemic, EyesDefyAnemia, EdgeInputHandler, FingernailAnemiaDataset

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

    # if dataset_type == "all":
    #     print(f"{tag} Processing dataset: {input_path}")
    #     handler = UnifiedAnemiaDataset(cp_dir="~/cp-anemia-detection/data/cp-anemic",
    #                            eyes_dir="~/cp-anemia-detection/data/eyes-defy-anemia",
    #                            transform=transform, tag="[Unified]")
    #     train_dataset, test_dataset = handler.get_datasets()
    #     train_loader, test_loader = handler.get_dataloaders(batch_size=batch_size)
    #     return [train_dataset, test_dataset], [train_loader, test_loader]

    if dataset_type == "edge-input":
        print(f"{tag} Processing edge input: {dataset_dir}")
        handler = EdgeInputHandler(image_path=dataset_dir, transform=edge_transform, tag=tag)
        dataloader = handler.get_dataloader(batch_size=batch_size)
        return dataloader
    
    elif dataset_type == "fingernail-anemia":
        print(f"{tag} Processing dataset: {dataset_dir}")
        handler = FingernailAnemiaDataset(data_dir=dataset_dir, transform=transform, tag=tag)
        train_dataset, test_dataset = handler.get_datasets()
        train_loader, test_loader = handler.get_dataloaders(batch_size=batch_size)
        return [train_dataset, test_dataset], [train_loader, test_loader]
    
    elif dataset_type == "cp-anemic":
        print(f"{tag} Processing dataset: {dataset_dir}")
        handler = CPAnemic(data_dir=dataset_dir, transform=transform, tag=tag)
        train_dataset, test_dataset = handler.get_datasets()
        train_loader, test_loader = handler.get_dataloaders(batch_size=batch_size)
        # import pdb; pdb.set_trace()
        return [train_dataset, test_dataset], [train_loader, test_loader]
    
    elif dataset_type == "eyes-defy-anemia":
        print(f"{tag} Processing dataset: {dataset_dir}")
        handler = EyesDefyAnemia(data_dir=dataset_dir, transform=transform, tag=tag)
        train_dataset, test_dataset = handler.get_datasets()
        train_loader, test_loader = handler.get_dataloaders(batch_size=batch_size)
        return [train_dataset, test_dataset], [train_loader, test_loader]
