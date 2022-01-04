from torch.utils.data import Dataset
import torch
import numpy as np
import os
import random
from PIL import Image
import torchvision.datasets as dset
from random import randrange
import random
import sys
sys.path.append('..')
from utils.util import generate_stroke_mask
import torchvision.transforms as transforms


# TODO: Add random crop and random flip [√]
def read_img(img_url, image_size, is_train):
    img = Image.open(img_url).convert("L")

    if random.random() > 0.5 and is_train:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


# TODO: mask data augmentation
class FaceScapeDataset(Dataset):
    
    def __init__(self, pt_dataset, mask_path=None, is_train=False, image_size=256, random_stroke=False):

        self.is_train = is_train
        self.pt_dataset = pt_dataset
        self.image_id_list = []

        temp_list = os.listdir(pt_dataset)
        for x in temp_list:
            if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.JPG') or x.endswith('.PNG'):
                self.image_id_list.append(x)

        self.random_stroke = random_stroke
        # self.perm = torch.arange(image_size*image_size)

        self.mask_dir = mask_path
        self.mask_list = os.listdir(self.mask_dir)
        self.mask_num = len(self.mask_list)
        self.image_size = image_size

        self.block_num = 32*32

        self.gray_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size), Image.BILINEAR),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        self.normalize = transforms.Normalize((0.5,), (0.5,))

        print("# Mask is %d, # Image is %d" % (self.mask_num, len(self.image_id_list)))
        
    def __len__(self):
        return len(self.image_id_list)

    def __getitem__(self, idx):

        if self.is_train:
            selected_mask_name = random.sample(self.mask_list, 1)[0]
        else:
            selected_mask_name = self.mask_list[idx % self.mask_num]

        # if not self.random_stroke:
        selected_mask_dir = os.path.join(self.mask_dir, selected_mask_name)
        # selected_mask_dir=selected_mask_name
        mask = Image.open(selected_mask_dir).convert("L")

        # else:
        #     mask = generate_stroke_mask([256, 256])
            # mask = (mask > 0).astype(np.uint8) * 255
            # mask = Image.fromarray(mask).convert("L")

        # if self.is_train:
        #     if random.random() > 0.5:
        #         mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        #     if random.random() > 0.5:
        #         mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        mask = self.gray_transforms(mask)

        mask[mask > 0.99] = 1
        mask[mask != 1] = 0

        # # mask = torch.from_numpy(np.array(mask)).view(-1)
        # mask = torch.from_numpy(np.array(mask))
        # mask = (mask/255.) > 0.5
        # mask = mask.float()

        selected_img_name = self.image_id_list[idx]
        selected_img_url = os.path.join(self.pt_dataset, selected_img_name)
        image = read_img(selected_img_url, image_size=self.image_size, is_train=self.is_train)

        image = self.gray_transforms(image)
        image = self.normalize(image)
        masked = mask * image

        # x = torch.from_numpy(np.array(x)).view(-1).float()  # flatten out all pixels
        # perm用于重排序，比如用一个倒序的列表就可以把data倒序掉，用这个写法可以很easy完成序列中的顺序调换
        # a[0]：取a序列的第一个元素，a[[0,1]]，按顺序取a序列的第1，2个元素
        # x = x[self.perm].float() # reshuffle pixels with any fixed permutation and -> float

        return masked, image, mask
