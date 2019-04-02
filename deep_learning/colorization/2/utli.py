import numpy as np
import torch
from os import walk
import torch.utils.data as Data
from torchvision import transforms,datasets
from skimage.color import rgb2lab, rgb2gray
from skimage import io
from PIL import Image

dir = '/home/wangzhihao/code/AI_new_man/colorization/VOCdevkit/VOC2012/JPEGImages/'
crop_transform = transforms.Compose([
    transforms.Scale(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
])
def getdir(a,b):
    image_list = []
    for (dirpath,dirname,filenames) in walk(dir):
        image_list.extend(filenames)
        break
    return image_list[a:b]


class ImageLoader(Data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, image_list, root_dir, transform):
        self.image_list = image_list
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image=Image.open(self.root_dir + self.image_list[idx])
        image = self.transform(image) # 3*224*224
        image_copy = image.clone()
        image = image.resize_(224,224,3) # 224*224*3
        image = image.numpy()
        image_gray = rgb2gray(image)  #224*224
        image_gray = torch.from_numpy(image_gray)
        image_gray = image_gray.unsqueeze(0) # 1*224*224
        image_gray = image_gray.float()
        image_copy = image_copy.float() #3*224*224
        return image_gray,image_copy



def dataloader(image_list,dir, m_batch_size):
    all_data = ImageLoader(image_list,dir,crop_transform)
    loader = Data.DataLoader(
        dataset=all_data,     
        batch_size = m_batch_size,     
        shuffle=True,              
        num_workers=1,             
    )
    return loader

