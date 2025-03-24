import os
import numpy as np
import torchvision.transforms as transforms
from utils.preprocess import DatasetHandler

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

cpanemic_path = os.path.abspath(os.path.join(os.getcwd(), "data/cp-anemia/"))
handler = DatasetHandler(data_dir=cpanemic_path,
                              transform=transform, tag="[ETL]")

train_loader, test_loader = handler.get_dataloaders(batch_size=1)
handler.save_to_json(cpanemic_path)