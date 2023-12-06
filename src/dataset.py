import os
from PIL import Image
import numpy as np
import scipy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from utils import HEDJitter

class UBCTilesDataset(Dataset):
    """UBC OCEAN tiles dataset."""

    def __init__(self, df, image_dir, transform=None):
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
        transforms.RandomAffine(degrees=CFG['affine_degrees'], translate=CFG['affine_translate'], scale=CFG['affine_scale']),
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
    def __init__(self, scale, fill=0):
        super().__init__()
        self.scale = scale
        self.fill = fill
    def forward(self, image):
        new_image = transforms.functional.affine(image, angle=0, translate=(0, 0), scale=self.scale, shear=0, fill=self.fill)
        return new_image
    
class SmartCrop(nn.Module):
    def split_crop(self, image, dir: str, th: int=0):
        if dir == 'h':
            labels, _ = scipy.ndimage.label(image.sum(axis=(0, 2)) > th)
        elif dir == 'v':
            labels, _ = scipy.ndimage.label(image.sum(axis=(1, 2)) > th)
        unique, counts = np.unique(labels, return_counts=True)
        counts = counts[unique!=0]
        unique = unique[unique!=0]
        larges_label = unique[np.argmax(counts)]
        if dir == 'h':
            image = image[:, labels==larges_label, :]
        elif dir == 'v':
            image = image[labels==larges_label, :, :]
        return image

    def forward(self, image):
        image = np.array(image)

        # Iterative split and crop
        image = self.split_crop(image, 'h')
        image = self.split_crop(image, 'v')
        image = self.split_crop(image, 'h')
        image = self.split_crop(image, 'v')

        black_bg = np.sum(image, axis=2) == 0
        image[black_bg, :] = 255

        return Image.fromarray(image)

def get_datasets(CFG, train_image_dir, train_thumbnail_dir, df_train, df_validation):
    transform = {
    'train':
    transforms.Compose([
        HEDJitter(theta=CFG['hed_theta']),
        SmartCrop(),
        transforms.RandomAffine(degrees=CFG['affine_degrees'], translate=CFG['affine_translate'], scale=CFG['affine_scale'], fill=255),
        transforms.Resize(CFG['img_size']),
        transforms.CenterCrop(CFG['img_size']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.ColorJitter(**CFG['color_jitter']),
        transforms.ToTensor(),
        # transforms.Normalize(mean=CFG['img_color_mean'], std=CFG['img_color_std'])
    ]),
    'validation':
     transforms.Compose([
        SmartCrop(),
        Affine(scale=np.mean(CFG['affine_scale']), fill=255),
        transforms.Resize(CFG['img_size']),
        transforms.CenterCrop(CFG['img_size']),
        transforms.ToTensor(),
        # transforms.Normalize(mean=CFG['img_color_mean'], std=CFG['img_color_std'])
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

# Tumor tile detection
class UBCTumorTilesDataset(Dataset):
    """UBC OCEAN tiles dataset."""

    def __init__(self, df, image_dir, transform=None):
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
        self.image_dir = image_dir
        self.transform = transform

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

def get_tumor_tiles_datasets(CFG, train_image_dir, df_train, df_validation):
    transform = {
    'train':
    transforms.Compose([
        transforms.Resize(CFG['img_size']),
        transforms.RandomAffine(degrees=CFG['affine_degrees'], translate=CFG['affine_translate'], scale=CFG['affine_scale']),
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
    
    train_dataset = UBCTumorTilesDataset(df=df_train, 
                        image_dir=train_image_dir,
                        transform=transform['train'])
    validation_dataset = UBCTumorTilesDataset(df=df_validation, 
                        image_dir=train_image_dir,
                        transform=transform['validation'])
    datasets = {'train': train_dataset, 'validation': validation_dataset}
    return datasets
