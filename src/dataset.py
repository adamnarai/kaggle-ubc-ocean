import os
from PIL import Image
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import HEDJitter
    
class UBCTilesDataset(Dataset):
    """UBC OCEAN tiles dataset."""

    def __init__(self, df, image_dir, transform=None, black_th=1.0):
        """
        Arguments:
            df (DataFrame): DataFrame of annotations.
            image_dir (string): Directory with all the fullres images.
            transform (callable, optional): Optional transform to be applied on a sample.
            black_th (float): Tiles with black > black_th proportion of black pixels are discarded.
        """
        self.df = df
        self.image_ids = self.df['image_id'].values
        self.orig_image_ids = self.df['orig_image_id'].values
        self.labels = self.df['label'].values
        self.is_tma = self.df['is_tma'].values
        self.image_dir = image_dir
        self.transform = transform
        self.black_th = black_th

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        orig_image_id = self.orig_image_ids[idx]
        image_path = os.path.join(self.image_dir, str(orig_image_id), f'{image_id}.png')

        image = Image.open(image_path)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label

def get_tiles_datasets(CFG, train_image_dir, df_train, df_validation):
    transform = {
    'train':
    transforms.Compose([
        transforms.Resize(CFG['img_size']),
        transforms.RandomAffine(degrees=CFG['affine_degrees'], translate=CFG['affine_translate'], scale=CFG['affine_scale'], fill=255),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CFG['img_color_mean'], std=CFG['img_color_std'])
    ]),
    'validation':
     transforms.Compose([
        transforms.Resize(CFG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=CFG['img_color_mean'], std=CFG['img_color_std'])
    ])}
    
    train_dataset = UBCTilesDataset(df=df_train, 
                        image_dir=train_image_dir,
                        transform=transform['train'])
    validation_dataset = UBCTilesDataset(df=df_validation, 
                        image_dir=train_image_dir,
                        transform=transform['validation'])
    datasets = {'train': train_dataset, 'validation': validation_dataset}
    return datasets

def get_dataloaders(CFG, datasets):
    train_loader = DataLoader(datasets['train'], batch_size=CFG['batch_size'], shuffle=True, num_workers=CFG['dataloader_num_workers'], pin_memory=True)
    validation_loader = DataLoader(datasets['validation'], batch_size=CFG['batch_size'], shuffle=False, num_workers=CFG['dataloader_num_workers'], pin_memory=True)
    dataloaders = {'train': train_loader, 'validation': validation_loader}
    return dataloaders


class UBCDataset(Dataset):
    """UBC OCEAN dataset."""

    def __init__(self, df, image_dir, thumbnail_dir, transform=None):
        """
        Arguments:
            df (DataFrame): DataFrame of annotations.
            image_dir (string): Directory with all the fullres images.
            thumbnail_dir (string): Directory with all the thumbnail images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.image_ids = self.df['image_id'].values
        self.labels = self.df['label'].values
        self.is_tma = self.df['is_tma'].values
        self.image_dir = image_dir
        self.thumbnail_dir = thumbnail_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.thumbnail_dir, f'{image_id}_thumbnail.png')
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, f'{image_id}.png')

        image = np.array(Image.open(image_path))
        label = self.labels[idx]
        
        if self.transform:
            try:
                image = self.transform(Image.fromarray(image))
            except(ValueError):
                print('Transform error')
                print(image_path)
                print(image.shape)

        return image, label
    
class SideCrop(nn.Module):
    def forward(self, image):
        size = min(image.size)
        new_image = transforms.functional.crop(image, 0, 0, size, size)
        return new_image
    
class Affine(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, image):
        new_image = transforms.functional.affine(image, angle=0, translate=(0, 0), scale=self.scale, shear=0, fill=255)
        return new_image
    
class SmartCrop(nn.Module):
    def forward(self, image):
        image = np.array(image)
        # Horizontal crop
        x_noblack = np.where(image.sum(axis=(0, 2))!=0)[0][0]
        image = image[:, x_noblack:, :]
        x_black = np.where(image.sum(axis=(0, 2))==0)[0]
        if x_black.size > 0 and x_black[0] > 512:
            x_end = x_black[0]-1
            image = image[:, :x_end, :]

        # Vertical crop and black trimming
        y_noblack = np.where(image.sum(axis=(1, 2))!=0)[0][0]
        image = image[y_noblack:, :, :]
        y_black = np.where(image.sum(axis=(1, 2))==0)[0]
        if y_black.size > 0 and y_black[0] > 400:
            y_end = y_black[0]-1
            image = image[:y_end, :, :]

        x_black = np.where(image.sum(axis=(0, 2))==0)[0]
        if x_black.size > 0 and x_black[0] > 400:
            x_end = x_black[0]-1
            image = image[:, :x_end, :]

        black_bg = np.sum(image, axis=2) == 0
        image[black_bg, :] = 255

        return Image.fromarray(image)

def get_datasets(CFG, train_image_dir, train_thumbnail_dir, df_train, df_validation):
    transform = {
    'train':
    transforms.Compose([
        SmartCrop(),
        transforms.RandomAffine(degrees=CFG['affine_degrees'], translate=CFG['affine_translate'], scale=CFG['affine_scale'], fill=255),
        transforms.Resize(CFG['img_size']),
        transforms.CenterCrop(CFG['img_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(**CFG['color_jitter']),
        # HEDJitter(theta=CFG['hed_theta']),
        transforms.ToTensor(),
        transforms.Normalize(mean=CFG['img_color_mean'], std=CFG['img_color_std'])
    ]),
    'validation':
     transforms.Compose([
        SmartCrop(),
        Affine(scale=1.2),
        transforms.Resize(CFG['img_size']),
        transforms.CenterCrop(CFG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=CFG['img_color_mean'], std=CFG['img_color_std'])
    ])}
    
    train_dataset = UBCDataset(df=df_train, 
                        image_dir=train_image_dir, 
                        thumbnail_dir=train_thumbnail_dir,
                        transform=transform['train'])
    validation_dataset = UBCDataset(df=df_validation, 
                        image_dir=train_image_dir, 
                        thumbnail_dir=train_thumbnail_dir,
                        transform=transform['validation'])
    datasets = {'train': train_dataset, 'validation': validation_dataset}
    return datasets
