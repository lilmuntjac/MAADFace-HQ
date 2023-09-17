from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

# -------------------- fairness related --------------------
# all the binary attribute
attributes47 = ['Male','Young','Middle_Aged','Senior','Asian','White','Black',
                'Rosy_Cheeks','Shiny_Skin','Bald','Wavy_Hair','Receding_Hairline','Bangs','Sideburns','Black_Hair','Blond_Hair','Brown_Hair','Gray_Hair',
                'No_Beard','Mustache','5_o_Clock_Shadow','Goatee','Oval_Face','Square_Face','Round_Face','Double_Chin','High_Cheekbones','Chubby',
                'Obstructed_Forehead','Fully_Visible_Forehead','Brown_Eyes','Bags_Under_Eyes','Bushy_Eyebrows','Arched_Eyebrows',
                'Mouth_Closed','Smiling','Big_Lips','Big_Nose','Pointy_Nose','Heavy_Makeup',
                'Wearing_Hat','Wearing_Earrings','Wearing_Necktie','Wearing_Lipstick','No_Eyewear','Eyeglasses','Attractive']

def split_tensor_by_attribute(tensor, attr_dim=0, attr_list=[]):
    # attribute list should contain the name of the attributes
    attr_idx = []
    for attr_name in attr_list:
        assert attributes47.count(attr_name) > 0, f'Unknown attribute'
        attr_idx.append(attributes47.index(attr_name))
    # filter tensor by the attributes
    match attr_dim:
        case 0:
            return tensor[attr_idx]
        case 1:
            return tensor[:, attr_idx]
        case 2:
            return tensor[:, :, attr_idx]
        case 3:
            return tensor[:, :, :, attr_idx]
        case _:
            assert False, f'only support attributes dimension less than 4'

def regroup_tensor_binary(tensor, sens, regroup_dim=0):
    # cannot distinguish negative label from the cell mask by sensitive value
    # the tenser need to be break up into 2
    match regroup_dim:
        case 0:
            group_1_tensor = tensor[sens[:,0]!=0]
            group_2_tensor = tensor[sens[:,0]==0]
        case 1:
            group_1_tensor = tensor[:,sens[:,0]!=0]
            group_2_tensor = tensor[:,sens[:,0]==0]
        case _:
            assert False, 'regroup dimension only support 0 and 1'
    return group_1_tensor, group_2_tensor

def divide_tensor_binary(tensor, divide_dim=0):
    # cut the tensor in half
    div = tensor.shape[divide_dim] // 2
    match divide_dim:
        case 0:
            group_1_tensor = tensor[0:div]
            group_2_tensor = tensor[div:]
        case 1:
            group_1_tensor = tensor[:,0:div]
            group_2_tensor = tensor[:,div:]
        case _:
            assert False, 'regroup dimension only support 0 and 1'
    return group_1_tensor, group_2_tensor

def regroup_tensor_categori(tensor, sens, regroup_dim=0):
    # sensitive attribute is race
    match regroup_dim:
        case 0:
            group_1_tensor = tensor[sens[:,0]==0]
            group_2_tensor = tensor[sens[:,0]!=0]
        case 1:
            group_1_tensor = tensor[:,sens[:,0]==0]
            group_2_tensor = tensor[:,sens[:,0]!=0]
        case _:
            assert False, 'regroup dimension only support 0 and 1'
    return group_1_tensor, group_2_tensor
            
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

def calc_groupacc(pred, label, sens):
    # assume the sensitive attribute is race
    group_1_pred, group_1_label = pred[sens[:,0]==0], label[sens[:,0]==0]
    group_2_pred, group_2_label = pred[sens[:,0]!=0], label[sens[:,0]!=0]
    group_1_result, group_2_result = torch.eq(group_1_pred, group_1_label), torch.eq(group_2_pred, group_2_label)
    group_1_correct, group_1_wrong = torch.sum(group_1_result, dim=0), torch.sum(torch.logical_not(group_1_result), dim=0)
    group_2_correct, group_2_wrong = torch.sum(group_2_result, dim=0), torch.sum(torch.logical_not(group_2_result), dim=0)
    stat = torch.stack((group_1_correct, group_1_wrong, group_2_correct, group_2_wrong), dim=1)
    return stat.detach().cpu().numpy()

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

def fairness_matrix_to_dict_key(fairness_matrix):
    match fairness_matrix:
        case "equality of opportunity":
            return "equality_of_opportunity" 
        case "equalized odds":
            return "equalized_odds"
        case "accuracy difference":
            return "acc_diff"
        case _:
            assert False, f'unrecognized fairness matrix' 
            
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
    return pred