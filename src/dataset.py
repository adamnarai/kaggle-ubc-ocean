import os
from PIL import Image

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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

        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(image_path)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label
    
class UBCTilesDataset(Dataset):
    """UBC OCEAN tiles dataset."""

    def __init__(self, df, image_dir, transform=None):
        """
        Arguments:
            df (DataFrame): DataFrame of annotations.
            image_dir (string): Directory with all the fullres images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
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
        image_path = os.path.join(self.image_dir, orig_image_id, f'{image_id}.png')

        image = Image.open(image_path)
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return image, label

class SideCrop(nn.Module):
    def forward(self, image):
        size = min(image.size)
        new_image = transforms.functional.crop(image, 0, 0, size, size)
        return new_image

def get_datasets(CFG, train_image_dir, train_thumbnail_dir, df_train, df_validation):
    transform = {
    'train':
    transforms.Compose([
        SideCrop(),
        transforms.Resize(CFG['img_size']),
        transforms.RandomAffine(degrees=CFG['affine_degrees'], translate=CFG['affine_translate'], scale=CFG['affine_scale']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'validation':
     transforms.Compose([
        SideCrop(),
        transforms.Resize(CFG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

def get_tiles_datasets(CFG, train_image_dir, df_train, df_validation):
    transform = {
    'train':
    transforms.Compose([
        SideCrop(),
        transforms.Resize(CFG['img_size']),
        transforms.RandomAffine(degrees=CFG['affine_degrees'], translate=CFG['affine_translate'], scale=CFG['affine_scale']),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'validation':
     transforms.Compose([
        SideCrop(),
        transforms.Resize(CFG['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    train_loader = DataLoader(datasets['train'], batch_size=CFG['batch_size'], shuffle=True, num_workers=36, pin_memory=True)
    validation_loader = DataLoader(datasets['validation'], batch_size=CFG['batch_size'], shuffle=False, num_workers=36, pin_memory=True)
    dataloaders = {'train': train_loader, 'validation': validation_loader}
    return dataloaders