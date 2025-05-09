from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import torch
from torchvision import transforms

class PreprocessData(Dataset):
    def __init__(self,root_dir, images_list, resize = (32,32)):
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # standard normalization values for ImageNet
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
    def __init__(self,root_dir, images_list, resize = (32,32)):
        self.root_dir = root_dir

        self.resize = resize
        self.mask_types = [
            'hair', 'skin', 
            'nose', 
            'l_eye','r_eye'
            ,'mouth'
        ]
        self.image_files = images_list
        self.transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # standard normalization values for ImageNet
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self,idx):
        image_name = self.image_files[idx][:-4]  # gets image name without .jpg
        
        image = Image.open(f"{self.root_dir}/images/{image_name}.jpg").convert('RGB')
        image = self.transform(image)
        
        
        masks = []
        for mask_type in self.mask_types:
            mask_path = f"{self.root_dir}/masks/{image_name}/{image_name}_{mask_type}.png"
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                mask = self.mask_transform(mask) > 0.5  # resize and binarize
            else:
                mask = torch.zeros(1, *self.resize)
            masks.append(mask)
        image_tensor = torch.tensor(image)
        masks_tensor = torch.cat(masks, dim=0)

        return image_tensor, masks_tensor
