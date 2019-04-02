import numpy as np
import pandas as pd
import torch
from torchvision.utils.data import DataLoader
from torchvision.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# default tansform resize to 256 and crop to 224(for 
# form alexnet to newest network the input is all 224 )
default_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
])

# here is your code to make the infor, it's a np and 0 is image file name 1 is label

def Process():
    pass




class Data(Dataset):
    """
    use Dataset to load data, you must overwrite the following functions

    Args:
        tansform: the transform add to your image
        root_dir: the image dir
        infor: a 2-D np, the 0 is image name and the 1 is label
    """

    def __init__(self, root_dir, infor, transform):
        super().__init__()
        self.root_dir = root_dir
        self.infor = infor
        self.tansform = transform

    def __len__(self):
        return len(self.infor)
    
    def __gititem(self, idx):
        # first get the image file
        image_file  = self.root_dir + self.infor[idx][0] + '.jpg'
        image = Image.opne(image_file)
        image = self.tansform(image)
        # attention if it's a classification task and you use cross entropy 
        # as your loss function, the label will be an int, for more information,
        # please go  http://pytorch.org/docs/master/nn.html#torch.nn.CrossEntropyLoss
        label = self.infor[idx][1]
        return  image, label


def Loader(root_dir, infor, batch_size, number_workers=0, transforms=default_transform):
    """
    use the DataLoader class to load data, and create batch.

    Args:
        root_dir: the image dir
        infor: a 2-D np, the 0 is image name and the 1 is label
        number_workers: the number of multithreading， default is 0
        transform: the transform to your image, default is resize to 256 and crop to 224

    """
    data = Data(root_dir, infor, transform)
    mini_batch = DataLoader(
        dataset = data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = number_workers,
    )
    return mini_batch 

    

