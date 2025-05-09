from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import torch
from torchvision import transforms

class PreprocessData(Dataset):
    def __init__(self,root_dir, images_list, resize = (64,64)):
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # WGAN-GP expects data in [-1, 1] range.
        ])
        self.root_dir = root_dir
        self.image_files = images_list

    # Preprocess test images to fit in model
    def preprocess(self):
        images = []
        for i in range(len(self.image_files)):
            image_name = self.image_files[i]           
            image = Image.open(f"{self.root_dir}/{image_name}").convert('RGB')
            image = self.transform(image)
            image_tensor = torch.tensor(image)
            images.append(image_tensor)
        return torch.stack(images)
        

class CelebALoad(Dataset):
    def __init__(self,root_dir, images_list, resize = (64,64)):
        self.root_dir = root_dir
        self.resize = resize
        self.image_files = images_list
        self.transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # WGAN-GP expects data in [-1, 1] range.
        ])

        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self,idx):
        image_name = self.image_files[idx]
        
        image = Image.open(f"{self.root_dir}/{image_name}").convert('RGB')
        image = self.transform(image)

        return image
