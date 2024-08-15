import os
import glob
import torch
from PIL import Image
from utils import *
from torchvision import transforms

class Seg(torch.utils.data.Dataset):
    def __init__(self, img_path, label_path, csv_path, mode='train',transform = None):
        super().__init__()
        self.mode = mode
        self.img_list = glob.glob(os.path.join(img_path,  '*.png'))
        self.label_list = glob.glob(os.path.join(label_path, '*.png'))
        self.label_colormap = get_label_colormap(csv_path)
        self.to_tensor = transforms.ToTensor()
        self.transforms = transform

    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')
        label = Image.open(self.label_list[index]).convert('RGB')
        if self.transforms:
            img = self.transforms(img).float()
        label = image2label(label, self.label_colormap)
        label = torch.from_numpy(label)

        return img, label

    def __len__(self):
        return len(self.img_list)