from dataclasses import dataclass, field
from typing import Tuple
from collections import OrderedDict
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

# -------------------- dataset --------------------
class MAADFaceHQDataset(torch.utils.data.Dataset):
    """Custom pytorch Dataset class for MAAD-Face-HQ dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        self.attributes = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.attribute_list = ['Filename','Identity','Male','Young','Middle_Aged','Senior','Asian','White','Black',
                               'Rosy_Cheeks','Shiny_Skin','Bald','Wavy_Hair','Receding_Hairline','Bangs','Sideburns','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair',
                               'No_Beard','Mustache','5_o_Clock_Shadow','Goatee','Oval_Face','Square_Face','Round_Face','Double_Chin','High_Cheekbones','Chubby',
                               'Obstructed_Forehead','Fully_Visible_Forehead','Brown_Eyes','Bags_Under_Eyes','Bushy_Eyebrows','Arched_Eyebrows',
                               'Mouth_Closed','Smiling','Big_Lips','Big_Nose','Pointy_Nose','Heavy_Makeup',
                               'Wearing_Hat','Wearing_Earrings','Wearing_Necktie','Wearing_Lipstick','No_Eyewear','Eyeglasses','Attractive']
    
    def __len__(self):
        return len(self.attributes.index)
    
    def __getitem__(self, index: int):
        X = Image.open(self.root_dir / self.attributes.iloc[index]['Filename'])
        if self.transform:
            X = self.transform(X)

        target = self.attributes.iloc[index][self.attribute_list[2:]] # binary attributes
        target = torch.tensor(target)
        return X, target

@dataclass
class MAADFaceHQ:
    """
    get train and test dataloader from MAAD-Face-HQ dataset
    attribute source: https://github.com/pterhoer/MAAD-Face
    image is from Vggface2-HQ: https://github.com/NNNNAI/VGGFace2-HQ
    """

    batch_size: int = 128
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
        train_dataset = MAADFaceHQDataset(self.train_csv, self.root+'/train', transform=train_transform)
        test_dataset = MAADFaceHQDataset(self.test_csv, self.root+'/test', transform=test_transform)
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=12, pin_memory=True, drop_last=True,)
        self.test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, 
            shuffle=True, num_workers=12, pin_memory=True, drop_last=True,)

# -------------------- model --------------------
class MAADFaceHQModel(torch.nn.Module):
    """
    MAADFace-HQ attribute prediction model,
    change the "out_feature" for number of attributes predicted.
        47: All attributes of MAADFace-HQ
    """

    def __init__(self, out_feature=46, weights='ResNet50_Weights.DEFAULT'):
        super(MAADFaceHQModel, self).__init__()
        self.model = models.resnet50(weights=weights)
        in_feature = self.model.fc.in_features
        self.model.fc = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_feature, out_feature)),
            ('sigm', nn.Sigmoid()),
        ]))

    def forward(self, x):
        """
        Input:
            x: Image of faces        (N, C, H, W)
        Output:
            z: attribute predictions (N, out_feature)
        """
        z = self.model(x)
        return z

# -------------------- fairness related --------------------
def calc_groupcm_soft(pred, label, sens):
    def confusion_matrix_soft(pred, label, idx):
        label_strong = torch.where(label>0.4, 1, 0)
        label_weak = torch.where(label<0.6, 0, 1)
        tp = torch.mul(pred[:,idx], label_strong[:,idx]).sum()
        fp = torch.mul(pred[:,idx], torch.sub(1, label_strong[:,idx])).sum()
        fn = torch.mul(torch.sub(1, pred[:,idx]), label_weak[:,idx]).sum()
        tn = torch.mul(torch.sub(1, pred[:,idx]), torch.sub(1, label_weak[:,idx])).sum()
        return tp, fp, fn, tn
    group_1_pred, group_1_label = pred[sens[:,0]==1], label[sens[:,0]==1]
    group_2_pred, group_2_label = pred[sens[:,0]==0], label[sens[:,0]==0]
    stat = np.array([])
    for idx in range(label.shape[-1]):
        group_1_tp, group_1_fp, group_1_fn, group_1_tn = confusion_matrix_soft(group_1_pred, group_1_label, idx)
        group_2_tp, group_2_fp, group_2_fn, group_2_tn = confusion_matrix_soft(group_2_pred, group_2_label, idx)
        row = np.array([[group_1_tp.item(), group_1_fp.item(), group_1_fn.item(), group_1_tn.item(), 
                         group_2_tp.item(), group_2_fp.item(), group_2_fn.item(), group_2_tn.item()]])
        stat =  np.concatenate((stat, row), axis=0) if len(stat) else row
    return stat

# -------------------- mist --------------------
# load and save model
def load_model(model, optimizer, scheduler, name, root_folder='/tmp2/npfe/model_checkpoint'):
    # Load the model weight, optimizer, and random states
    folder = Path(root_folder)
    path = folder / f"{name}.pth"
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    torch.set_rng_state(ckpt['rng_state'])
    torch.cuda.set_rng_state(ckpt['cuda_rng_state'])

def save_model(model, optimizer, scheduler, name, root_folder='/tmp2/npfe/model_checkpoint'):
    # Save the model weight, optimizer, scheduler, and random states
    # create the root folder if not exist
    folder = Path(root_folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.pth"
    # save the model checkpoint
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
    }
    torch.save(save_dict, path)

# load and save stats
def load_stats(name, root_folder='/tmp2/npfe/model_stats'):
    # Load the numpy array
    folder = Path(root_folder)
    path = folder / f"{name}.npy"
    nparray = np.load(path)
    return nparray

def save_stats(nparray, name, root_folder='/tmp2/npfe/model_stats'):
    # Save the numpy array
    # create the root folder if not exist
    folder = Path(root_folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.npy"
    np.save(path, nparray)

# normalize
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std  = (0.229, 0.224, 0.225)
def normalize(data, mean=imagenet_mean, std=imagenet_std):
    # Normalize batch of images
    transform = transforms.Normalize(mean=mean, std=std)
    return transform(data)

# to_prediction
def to_prediction(logit):
    # conert binary logit into prediction
    pred = torch.where(logit > 0.5, 1, 0)