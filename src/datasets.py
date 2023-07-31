from dataclasses import dataclass, field
from typing import List, Tuple
from collections import OrderedDict
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

#
# -------------------- MAADFace-HQ dataset --------------------
#
class MAADFaceHQDataset(torch.utils.data.Dataset):
    """Custom pytorch Dataset class for MAAD-Face-HQ dataset"""

    def __init__(self, csv_file, root_dir, attr_list=[], transform=None):
        self.attributes = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.attribute_list = ['Filename','Identity','Male','Young','Middle_Aged','Senior','Asian','White','Black',
                               'Rosy_Cheeks','Shiny_Skin','Bald','Wavy_Hair','Receding_Hairline','Bangs','Sideburns','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair',
                               'No_Beard','Mustache','5_o_Clock_Shadow','Goatee','Oval_Face','Square_Face','Round_Face','Double_Chin','High_Cheekbones','Chubby',
                               'Obstructed_Forehead','Fully_Visible_Forehead','Brown_Eyes','Bags_Under_Eyes','Bushy_Eyebrows','Arched_Eyebrows',
                               'Mouth_Closed','Smiling','Big_Lips','Big_Nose','Pointy_Nose','Heavy_Makeup',
                               'Wearing_Hat','Wearing_Earrings','Wearing_Necktie','Wearing_Lipstick','No_Eyewear','Eyeglasses','Attractive']
        if  len(attr_list) == 0:
            self.filter = [*range(2,49)] # binary attributes
        else:
            self.filter = []
            for attr_name in attr_list:
                assert self.attribute_list.count(attr_name) > 0, f'Unknown attribute'
                self.filter.append(self.attribute_list.index(attr_name))
    
    def __len__(self):
        return len(self.attributes.index)
    
    def __getitem__(self, index: int):
        X = Image.open(self.root_dir / self.attributes.iloc[index]['Filename'])
        if self.transform:
            X = self.transform(X)

        target = self.attributes.iloc[index][self.filter]
        target = torch.tensor(target)
        target = torch.div(target + 1, 2) # map from {-1, 1} to {0, 1}
        return X, target

@dataclass
class MAADFaceHQ:
    """
    get train and test dataloader from MAAD-Face-HQ dataset
    attribute source: https://github.com/pterhoer/MAAD-Face
    image is from Vggface2-HQ: https://github.com/NNNNAI/VGGFace2-HQ
    """

    batch_size: int = 128
    attr_list: list[str] = field(default_factory=list)
    root: str = '/tmp2/dataset/MAADFace_HQ'
    train_csv: str = '/tmp2/dataset/MAADFace_HQ/MAADFace_HQ_train.csv'
    test_csv: str = '/tmp2/dataset/MAADFace_HQ/MAADFace_HQ_test.csv'
    mean: Tuple = (0.485, 0.456, 0.406)  # same as ImageNet
    std: Tuple = (0.229, 0.224, 0.225)

    train_dataloader: torch.utils.data.DataLoader = field(init=False)
    val_dataloader: torch.utils.data.DataLoader = field(init=False)

    def __post_init__(self):
        train_transform: transforms.transforms.Compose = \
            transforms.Compose([
                transforms.TrivialAugmentWide(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # To do extra operation onto the image
                # we decided to normalize the image right before passing it into the model
                # transforms.Normalize(mean, std),
                # because of this, random erasing pixel as imagenet mean, not zero
                # transforms.RandomErasing(0.2, value=self.mean),
            ])
        test_transform: transforms.transforms.Compose = \
            transforms.Compose([
                transforms.Resize((224, 224)), 
                transforms.ToTensor(),
                # To do extra operation onto the image
                # we decided to normalize the image right before passing it into the model
                # transforms.Normalize(mean, std),
            ])
        # select the dataset source
        train_dataset = MAADFaceHQDataset(self.train_csv, self.root+'/train', 
            attr_list=self.attr_list, transform=train_transform)
        test_dataset = MAADFaceHQDataset(self.test_csv, self.root+'/test', 
            attr_list=self.attr_list, transform=test_transform)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=20, pin_memory=True, drop_last=True,)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=20, pin_memory=True, drop_last=True,)

    

#
# -------------------- CelebA dataset --------------------
#
from torchvision.datasets import CelebA
from typing import Callable, List, Optional, Union
class CelebADataset(CelebA):
    """edit from torchvision CelebA to allow different folder name"""

    def __init__(
        self,
        root: str, # not the root the image download to. it's the folder itself
        split: str = "train",
        attr_list: List[str] = [],
        target_type: Union[List[str], str] = "attr",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ):
        base_path = Path(root)
        self.base_folder = base_path.name
        root_path = base_path.parents[0]
        super().__init__(
            root_path,
            split,
            target_type,
            transform,
            target_transform,
            download
        )
        self.attribute_list = ['5_o_Clock_Shadow','Arched_Eyebrows','Attractive','Bags_Under_Eyes','Bald','Bangs',
                               'Big_Lips','Big_Nose','Black_Hair','Blond_Hair','Blurry','Brown_Hair',
                               'Bushy_Eyebrows','Chubby','Double_Chin','Eyeglasses','Goatee','Gray_Hair',
                               'Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open','Mustache',
                               'Narrow_Eyes','No_Beard','Oval_Face','Pale_Skin','Pointy_Nose','Receding_Hairline',
                               'Rosy_Cheeks','Sideburns','Smiling','Straight_Hair','Wavy_Hair','Wearing_Earrings',
                               'Wearing_Hat','Wearing_Lipstick','Wearing_Necklace','Wearing_Necktie','Young']
        if len(attr_list) == 0:
            self.filter = [*(range(40))]
        else:
            self.filter = []
            for attr_name in attr_list:
                assert self.attribute_list.count(attr_name) > 0, f'Unknown attribute'
                self.filter.append(self.attribute_list.index(attr_name))

    def __getitem__(self, index: int):
        X, target = super().__getitem__(index)
        target = target[self.filter]
        return X, target

@dataclass
class CelebA:
    """get train and vaildation dataloader from CelebA dataset"""

    batch_size: int = 128
    attr_list: list[str] = field(default_factory=list)
    root: str = '/tmp2/dataset/celeba'
    mean: Tuple = (0.485, 0.456, 0.406)
    std: Tuple = (0.229, 0.224, 0.225)

    train_dataloader: torch.utils.data.DataLoader = field(init=False)
    val_dataloader: torch.utils.data.DataLoader = field(init=False)

    def __post_init__(self):
        train_transform = transforms.Compose([
            transforms.TrivialAugmentWide(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # To do extra operation onto the image
            # we decided to normalize the image right before passing it into the model
            # transforms.Normalize(mean, std),
            # because of this, random erasing pixel as imagenet mean, not zero
            # transforms.RandomErasing(0.2, value=self.mean),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            # To do extra operation onto the image
            # we decided to normalize the image right before passing it into the model
            # transforms.Normalize(mean, std),
        ])
        # dataset source from pytorch official
        train_dataset = CelebADataset(root=self.root, split='train', target_type='attr', 
            attr_list=self.attr_list, transform=train_transform, download=False)
        val_dataset = CelebADataset(root=self.root, split='valid', target_type='attr', 
            attr_list=self.attr_list, transform=val_transform, download=False)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=20, pin_memory=True, drop_last=True,)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=20, pin_memory=True, drop_last=True,)
