from torch.utils.data import Dataset
import torch
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        super().__init__()
        self.root_dir = root_dir
        self.transforms=transforms
        self.images_list = os.listdir(root_dir)
        self.images_list = self.images_list[0:1500]

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
            
        image_path = os.path.join(self.root_dir, self.images_list[index])
        image = Image.open(image_path)
        
        if self.transforms:
            image = self.transforms(image)
            
        return image