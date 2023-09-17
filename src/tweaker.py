from dataclasses import dataclass, field
from typing import Tuple
from collections import OrderedDict
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import dlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from .perturbations import *
from .utils import *

@dataclass
class Tweaker:
    tweak_type: str

    batch_size: int = 128
    width: int = 224
    device: str = 'cuda'

    noise_budget: float = 0.0625
    circle_sharpness: float = 40.
    circle_rotation: float = 0.083
    circle_ratio: tuple[float, float] = (0.02, 0.03)
    frame_thickness: float = 0.25
    face_detector: dlib.cnn_face_detection_model_v1 = \
        dlib.cnn_face_detection_model_v1('./dlib_models/mmod_human_face_detector.dat')
    shape_predictor: dlib.shape_predictor = \
        dlib.shape_predictor('./dlib_models/shape_predictor_68_face_landmarks.dat')

    # create after the width is defined
    coord_ref: list[float] = field(default_factory=list, init=False)
    # create after the tweak type is defined
    mask: torch.Tensor = field(init=False)
    def __post_init__(self):
        # init coord_ref
        coord_ref = np.linspace(-3.0, 3.0, num=6*self.width+1)
        coord_ref = coord_ref[1::2] # remove the odd index (edge between pixels)
        coord_ref = np.around(coord_ref, 4)
        # note: true_coord_ref = coord_ref[pixel_coord+width]
        self.coord_ref = coord_ref

        # init mask
        self.mask = self.get_mask()
        self.mask = self.mask.to(self.device)
        
    def get_mask(self):
        match self.tweak_type:
            case 'noise':
                mask = torch.ones(1, 3, self.width, self.width) # don't need a mask
            case 'patch':
                diameter = self.width
                x = torch.linspace(-1, 1, diameter)
                y = torch.linspace(-1, 1, diameter)
                xx, yy = torch.meshgrid(x, y, indexing='ij')
                z = (xx**2 + yy**2) ** self.circle_sharpness
                mask = 1 - np.clip(z, -1, 1)
                mask = mask.unsqueeze(0)
                mask = torch.cat((mask, mask, mask), 0)
                mask = mask.unsqueeze(0)
            case 'frame':
                base = torch.zeros(self.width, self.width)
                gap = self.frame_thickness*self.width/2
                for i in range(self.width):
                    for j in range(self.width):
                        if i < gap or j < gap:
                            base[i][j] = 1
                        if self.width-i < gap or self.width-j < gap:
                            base[i][j] = 1
                mask = base.unsqueeze(0)
                mask = torch.cat((mask, mask, mask), 0)
                mask  = mask.unsqueeze(0)
            case 'eyeglasses':
                image = Image.open('./eyeglasses_mask_6percent.png')
                PIL_to_tensor = transforms.ToTensor()
                mask = PIL_to_tensor(image)
                mask = mask.unsqueeze(0)
            case _:
                assert False, 'the tweak type is not supported.'
        return mask
        
    # -------------------- noise --------------------
    def add(self, x, element):
        x = x + element
        x = torch.clamp(x, min=0.0, max=1.0) # valid image range
        return x
    
    # -------------------- patch --------------------
    def get_circle_transform(self):
        theta = torch.empty(0)
        # create one transformation matrix at a time
        for b in range(self.batch_size):
            # rotation and scaling
            rot = (-2*self.circle_rotation)*torch.rand(1) + self.circle_rotation
            rot_matrix = torch.tensor(
                [[torch.cos(-rot), -torch.sin(-rot)],
                 [torch.sin(-rot), torch.cos(-rot)]]
            )
            scale = map(lambda x : 2*np.sqrt(x/np.pi), self.circle_ratio)
            scale_min, scale_max = scale
            scale = (scale_min-scale_max)*torch.rand(1) + scale_max
            inv_scale = 1.0 / scale
            scale_matrix = torch.tensor(
                [[inv_scale, 0],
                 [0, inv_scale]]
            )
            xform_matrix = torch.mm(rot_matrix, scale_matrix)
            # translation
            avoid_from_center = 0.5
            range_min, range_max = avoid_from_center+scale, 1-scale
            if range_min >= range_max:
                print(f'range min: {range_min}, range max: {range_max}')
                assert False, f'Patch is too large (or too close) to avoid the center of the image.'
            # keep trying until it fit
            while True:
                rnd_min, rnd_max = -(1-scale), 1-scale
                shift_x, shift_y = (rnd_min-rnd_max)*torch.rand(2) + rnd_max
                if abs(shift_x) >= range_min or abs(shift_y) >= range_min:
                    break
            shift_x, shift_y = shift_x*inv_scale, shift_y*inv_scale
            # if scale <= 1.0:
            #     shift_min, shift_max = -(1-scale)/scale, (1-scale)/scale
            # else:
            #     shift_min, shift_max = 0.0, 0.0
            # shift_x, shift_y = (shift_min-shift_max)*torch.rand(2) + shift_max
            xform_matrix = torch.cat((xform_matrix, torch.tensor([[shift_x], [shift_y]])), 1)
            xform_matrix = xform_matrix.unsqueeze(0)
            theta = torch.cat((theta, xform_matrix), dim=0) if len(theta) else xform_matrix
        return theta.to(self.device)
    
    # -------------------- frame --------------------
    def get_identity_transform(self):
        theta = torch.tensor([[[1.,0.,0.],[0.,1.,0.]]])
        theta = theta.repeat(self.batch_size, 1, 1)
        return theta.to(self.device)
    
    # -------------------- eyeglesses --------------------
    def get_landmark(self, data, label):
        # turn the image back to [0, 255] for face detector
        rgb_image = data.clone().detach().cpu().permute(0, 2, 3, 1).numpy()*255.9999
        rgb_image = rgb_image.astype(np.uint8)
        detectable, landmark = list(), list()
        for idx in range(self.batch_size):
            detected_face = self.face_detector(rgb_image[idx], 1)
            if len(detected_face) != 1:
                continue # only 1 face is allowed
            landmark.append(self.shape_predictor(rgb_image[idx], detected_face[0].rect))
            detectable.append(idx)
        filtered_data = data[detectable, :, :, :]
        filtered_label = label[detectable,]
        return filtered_data, filtered_label, landmark
    
    def get_torch_coord(self, point_list):
        new_coord = list()
        for point in point_list:
            x, y = int(point[0]), int(point[1])
            # landmark upper bound & lower bound
            new_x, new_y = self.coord_ref[x+self.width], self.coord_ref[y+self.width]
            new_coord.append([new_x, new_y])
        return new_coord

    def set_eyeglasses_transform(self, landmark, reference=[[73,75],[149,75],[111,130]]):
        reference = self.get_torch_coord(reference)
        theta = torch.empty(0)
        for lm in landmark:
            # get the transformed points from the landmark
            left_eye, right_eye, noise_tip = (lm.part(36)+lm.part(39))/2 ,(lm.part(42)+lm.part(45))/2, lm.part(33)
            destination = [[left_eye.x, left_eye.y], [right_eye.x, right_eye.y], [noise_tip.x, noise_tip.y]]
            destination = self.get_torch_coord(destination)
            for point in destination:
                point.append(1)
            destination = torch.tensor(destination, dtype=torch.float)
            outset = torch.tensor(reference, dtype=torch.float)
            xform_matrix = torch.linalg.solve(destination, outset).transpose(0,1)
            xform_matrix = xform_matrix.unsqueeze(0)
            theta = torch.cat((theta, xform_matrix), dim=0) if len(theta) else xform_matrix
        return theta.to(self.device)

    # --------------------  --------------------
    def apply(self, data, label, element):
        match self.tweak_type:
            case 'noise':
                tweak_data = self.add(data, element)
            case 'patch':
                theta = self.get_circle_transform()
                mask = self.mask.repeat(data.shape[0], 1, 1, 1)
                element = element.repeat(data.shape[0], 1, 1, 1)
                grid = F.affine_grid(theta, data.shape, align_corners=False)
                # xform_mask = F.grid_sample(mask, grid, align_corners=False)
                xform_component = F.grid_sample(element, grid, mode='bilinear', align_corners=False)
                # inv_mask = 1 - xform_mask
                # tweak_data = data*inv_mask + xform_component*xform_mask
                # new ver.
                tweak_data = data + xform_component
                tweak_data = torch.clamp(tweak_data, min=0.0, max=1.0)
            case 'frame':
                theta = self.get_identity_transform()
                mask = self.mask.repeat(data.shape[0], 1, 1, 1)
                element = element.repeat(data.shape[0], 1, 1, 1)
                grid = F.affine_grid(theta, data.shape, align_corners=False)
                xform_mask = F.grid_sample(mask, grid, align_corners=False)
                xform_component = F.grid_sample(element, grid, mode='bilinear', align_corners=False)
                inv_mask = 1 - xform_mask
                tweak_data = data*inv_mask + xform_component*xform_mask
            case 'eyeglasses':
                # so far only this method remove some data that fail on face detector
                filtered_data, filtered_label, landmark = self.get_landmark(data, label)
                theta = self.set_eyeglasses_transform(landmark)
                mask = self.mask.repeat(filtered_data.shape[0], 1, 1, 1)
                element = element.repeat(filtered_data.shape[0], 1, 1, 1)
                grid = F.affine_grid(theta, filtered_data.shape, align_corners=False)
                xform_mask = F.grid_sample(mask, grid, align_corners=False)
                xform_component = F.grid_sample(element, grid, mode='bilinear', align_corners=False)
                inv_mask = 1 - xform_mask
                tweak_data = filtered_data*inv_mask + xform_component*xform_mask
                label = filtered_label
            case _:
                assert False, 'the tweak type is not supported.'
        return tweak_data, label
    
    def retify(self, element):
        match self.tweak_type:
            case 'noise':
                element.data.clamp_(-self.noise_budget, self.noise_budget)
            case 'patch' | 'frame' | 'eyeglasses':
                element.data.clamp_(0., 1.)
            case _:
                assert False, 'the tweak type is not supported.'

        
@dataclass
class Losses:
    loss_type: str
    fairness_criteria: str
    pred_type: str
    out_feature: int = 0
    soft_label: bool = False

    # -------------------- cell mask --------------------
    def tp_cells(self, pred, label):
        if self.soft_label:
            label = torch.where(label>0.4, 1, 0) # strong positive label
        return torch.mul(pred, label)
    def fp_cells(self, pred, label):
        if self.soft_label:
            label = torch.where(label>0.4, 1, 0) # strong positive label
        return torch.mul(pred, torch.sub(1, label))
    def fn_cells(self, pred, label):
        if self.soft_label:
            label = torch.where(label<0.6, 0, 1) # strong negative label
        return torch.mul(torch.sub(1, pred), label)
    def tn_cells(self, pred, label):
        if self.soft_label:
            label = torch.where(label<0.6, 0, 1) # strong negative label
        return torch.mul(torch.sub(1, pred), torch.sub(1, label))
    
    # -------------------- cm ratio --------------------
    def get_tpr(self, pred, label, batch_dim=0):
        # return 1 under the division by zero scenario
        # return tpr per attributes
        numerator = torch.sum(self.tp_cells(pred, label), dim=batch_dim) # TP
        if self.soft_label:
            label = torch.where(label>0.4, 1, 0) # strong positive label
        denominator = torch.sum(label, dim=batch_dim) # all positive label
        tpr = torch.full_like(denominator, fill_value=1.0, dtype=torch.float)
        tpr_mask = (denominator != 0)
        tpr[tpr_mask] = numerator[tpr_mask]/denominator[tpr_mask]
        return tpr
    def get_tnr(self, pred, label, batch_dim=0):
        # return 1 under the division by zero scenario
        # return tnr per attributes
        numerator = torch.sum(self.tn_cells(pred, label), dim=batch_dim) # TN
        if self.soft_label:
            label = torch.where(label<0.6, 0, 1) # strong negative label
        denominator = torch.sum(torch.sub(1, label), dim=batch_dim) # all negative label
        tnr = torch.full_like(denominator, fill_value=1.0, dtype=torch.float)
        tnr_mask = (denominator != 0)
        tnr[tnr_mask] = numerator[tnr_mask]/denominator[tnr_mask]
        return tnr
    def get_fnr(self, pred, label, batch_dim=0):
        numerator = torch.sum(self.fn_cells(pred, label), dim=batch_dim) # FN
        if self.soft_label:
            label = torch.where(label>0.4, 1, 0) # strong positive label
        denominator = torch.sum(label, dim=batch_dim) # all positive label
        fnr = torch.full_like(denominator, fill_value=1.0, dtype=torch.float)
        fnr_mask = (denominator != 0)
        fnr[fnr_mask] = numerator[fnr_mask]/denominator[fnr_mask]
        return fnr
    def get_fpr(self, pred, label, batch_dim=0):
        numerator = torch.sum(self.fp_cells(pred, label), dim=batch_dim) # FP
        if self.soft_label:
            label = torch.where(label<0.6, 0, 1) # strong negative label
        denominator = torch.sum(torch.sub(1, label), dim=batch_dim) # all negative label
        fpr = torch.full_like(denominator, fill_value=1.0, dtype=torch.float)
        fpr_mask = (denominator != 0)
        fpr[fpr_mask] = numerator[fpr_mask]/denominator[fpr_mask]
        return fpr
    def get_acc(self, pred, label, batch_dim=0):
        numerator = torch.sum(self.tp_cells(pred, label), dim=batch_dim)+torch.sum(self.tn_cells(pred, label), dim=batch_dim) # TP+TN
        denominator = label.shape[batch_dim]
        accuracy = numerator/denominator
        return accuracy

    # -------------------- losses (binary) --------------------
    def get_bce_by_cells(self, logit, label, cells=['tp', 'fn', 'fp', 'tn']):
        # get the binary cross entropy loss per attributes with specific cm cells
        # get cells type
        pred = torch.where(logit> 0.5, 1, 0)
        cell_mask = torch.full_like(logit, fill_value=0.)
        for type in cells:
            match type:
                case 'tp':
                    cell_mask += self.tp_cells(pred, label)
                case 'fn':
                    cell_mask += self.fn_cells(pred, label)
                case 'fp':
                    cell_mask += self.fp_cells(pred, label)
                case 'tn':
                    cell_mask += self.tn_cells(pred, label)
                case _:
                    assert False, 'only "tp", "fn", "fp", "tn" are allowed'
        # mask with the bce loss
        bce_per_ins = F.binary_cross_entropy(logit, label, reduction='none')
        return torch.mul(bce_per_ins, cell_mask)

    def binary_direct_loss(self, logit, label, sens):
        # approximate prediction using a very steep function
        pred = 1./(1+torch.exp(-1e4*logit-0.5))
        group_1_pred, group_2_pred = regroup_tensor_binary(pred, sens, regroup_dim=0)
        group_1_label, group_2_label = regroup_tensor_binary(label, sens, regroup_dim=0)
        match self.fairness_criteria:
            case 'equality of opportunity':
                group_1_tpr = self.get_tpr(group_1_pred, group_1_label)
                group_2_tpr = self.get_tpr(group_2_pred, group_2_label)
                loss_per_attr = torch.abs(group_1_tpr-group_2_tpr)
            case 'equalized odds':
                group_1_tpr = self.get_tpr(group_1_pred, group_1_label)
                group_2_tpr = self.get_tpr(group_2_pred, group_2_label)
                group_1_tnr = self.get_tnr(group_1_pred, group_1_label)
                group_2_tnr = self.get_tnr(group_2_pred, group_2_label)
                loss_per_attr = torch.abs(group_1_tpr-group_2_tpr) + torch.abs(group_1_tnr-group_2_tnr)
            case _:
                assert False, f'unrecognized fairness criteria'
        return torch.mean(loss_per_attr)
    
    def binary_masking_loss(self, logit, label, sens):
        # find the relation between groups
        pred = 1./(1+torch.exp(-1e4*logit-0.5))
        group_1_pred, group_2_pred = regroup_tensor_binary(pred, sens, regroup_dim=0)
        group_1_label, group_2_label = regroup_tensor_binary(label, sens, regroup_dim=0)
        match self.fairness_criteria:
            case 'equality of opportunity':
                group_1_tpr = self.get_tpr(group_1_pred, group_1_label)
                group_2_tpr = self.get_tpr(group_2_pred, group_2_label)
                group_1_tpr_boost = torch.where(group_1_tpr < group_2_tpr, 1, 0)
                group_2_tpr_boost = 1-group_1_tpr_boost
                tp_bce = self.get_bce_by_cells(logit, label, ['tp',])
                group_1_tp_bce, group_2_tp_bce = regroup_tensor_binary(tp_bce, sens, regroup_dim=0)
                loss_per_attr = torch.sum(group_1_tp_bce*group_1_tpr_boost, dim=0)+\
                                torch.sum(group_2_tpr_boost*group_2_tp_bce, dim=0)
            case 'equalized odds':
                group_1_tpr = self.get_tpr(group_1_pred, group_1_label)
                group_2_tpr = self.get_tpr(group_2_pred, group_2_label)
                group_1_tnr = self.get_tnr(group_1_pred, group_1_label)
                group_2_tnr = self.get_tnr(group_2_pred, group_2_label)
                group_1_tpr_boost = torch.where(group_1_tpr < group_2_tpr, 1, 0)
                group_2_tpr_boost = 1-group_1_tpr_boost
                group_1_tnr_boost = torch.where(group_1_tnr < group_2_tnr, 1, 0)
                group_2_tnr_boost = 1-group_1_tnr_boost
                tp_bce = self.get_bce_by_cells(logit, label, ['tp',])
                tn_bce = self.get_bce_by_cells(logit, label, ['tn',])
                group_1_tp_bce, group_2_tp_bce = regroup_tensor_binary(tp_bce, sens, regroup_dim=0)
                group_1_tn_bce, group_2_tn_bce = regroup_tensor_binary(tn_bce, sens, regroup_dim=0)
                loss_per_attr = torch.sum(group_1_tp_bce*group_1_tpr_boost, dim=0)+ \
                                torch.sum(group_2_tpr_boost*group_2_tp_bce, dim=0)+ \
                                torch.sum(group_1_tn_bce*group_1_tnr_boost, dim=0)+ \
                                torch.sum(group_2_tnr_boost*group_2_tn_bce, dim=0)
            case _:
                assert False, f'unrecognized fairness criteria'
        return torch.mean(loss_per_attr)

    def binary_perturb_loss(self, logit, label, sens):
        # fairness function
        def perturbed_eqopp(x, label=label, sens=sens):
            pred = torch.where(x> 0.5, 1, 0)
            label_duped = label.repeat(x.shape[0], 1, 1)
            group_1_pred, group_2_pred = regroup_tensor_binary(pred, sens, regroup_dim=1)
            group_1_label, group_2_label = regroup_tensor_binary(label_duped, sens, regroup_dim=1)
            # group_1_pred, group_2_pred = divide_tensor_binary(pred, divide_dim=1)
            # group_1_label, group_2_label = divide_tensor_binary(label_duped, divide_dim=1)
            group_1_tpr = self.get_tpr(group_1_pred, group_1_label, batch_dim=1)
            group_2_tpr = self.get_tpr(group_2_pred, group_2_label, batch_dim=1)
            return torch.abs(group_1_tpr-group_2_tpr)
        def perturbed_eqodd(x, label=label, sens=sens):
            pred = torch.where(x> 0.5, 1, 0)
            label_duped = label.repeat(x.shape[0], 1, 1)
            group_1_pred, group_2_pred = regroup_tensor_binary(pred, sens, regroup_dim=1)
            group_1_label, group_2_label = regroup_tensor_binary(label_duped, sens, regroup_dim=1)
            # group_1_pred, group_2_pred = divide_tensor_binary(pred, divide_dim=1)
            # group_1_label, group_2_label = divide_tensor_binary(label_duped, divide_dim=1)
            group_1_tpr = self.get_tpr(group_1_pred, group_1_label, batch_dim=1)
            group_2_tpr = self.get_tpr(group_2_pred, group_2_label, batch_dim=1)
            group_1_tnr = self.get_tnr(group_1_pred, group_1_label, batch_dim=1)
            group_2_tnr = self.get_tnr(group_2_pred, group_2_label, batch_dim=1)
            return torch.abs(group_1_tpr-group_2_tpr)+torch.abs(group_1_tnr-group_2_tnr)
        match self.fairness_criteria:
            case 'equality of opportunity':
                pret_eqopp = perturbed(perturbed_eqopp, 
                                       num_samples=10000,
                                       sigma=0.5,
                                       noise='gumbel',
                                       batched=False)
                loss_per_attr = pret_eqopp(logit)
            case 'equalized odds':
                pret_eqodd = perturbed(perturbed_eqodd, 
                                       num_samples=10000,
                                       sigma=0.5,
                                       noise='gumbel',
                                       batched=False)
                loss_per_attr = pret_eqodd(logit)
            case _:
                assert False, f'unrecognized fairness criteria'
        return torch.mean(loss_per_attr)

    def binary_accuracy_perturb_loss(self, logit, label, sens, p_coef, n_coef):
        def perturbed_FNR(x, label=label, sens=sens):
            pred = torch.where(x> 0.5, 1, 0)
            # dupe the label to have the same shape as x
            label_duped = label.repeat(x.shape[0], 1, 1)
            # regroup and compute FNR for both groups
            group_1_pred, group_2_pred = regroup_tensor_binary(pred, sens, regroup_dim=1)
            group_1_label, group_2_label = regroup_tensor_binary(label_duped, sens, regroup_dim=1)
            group_1_fnr = self.get_fnr(group_1_pred, group_1_label, batch_dim=1)
            group_2_fnr = self.get_fnr(group_2_pred, group_2_label, batch_dim=1)
            return group_1_fnr+group_2_fnr
        def perturbed_FPR(x, label=label, sens=sens):
            pred = torch.where(x> 0.5, 1, 0)
            # dupe the label to have the same shape as x
            label_duped = label.repeat(x.shape[0], 1, 1)
            # regroup and compute FNR for both groups
            group_1_pred, group_2_pred = regroup_tensor_binary(pred, sens, regroup_dim=1)
            group_1_label, group_2_label = regroup_tensor_binary(label_duped, sens, regroup_dim=1)
            group_1_fpr = self.get_fpr(group_1_pred, group_1_label, batch_dim=1)
            group_2_fpr = self.get_fpr(group_2_pred, group_2_label, batch_dim=1)
            return group_1_fpr+group_2_fpr
        
        pret_fnr = perturbed(perturbed_FNR,
                             num_samples=10000,
                             sigma=0.5,
                             noise='gumbel',
                             batched=False)
        pret_fpr = perturbed(perturbed_FPR,
                             num_samples=10000,
                             sigma=0.5,
                             noise='gumbel',
                             batched=False)

        match self.fairness_criteria:
            case 'equality of opportunity':
                loss_per_attr = pret_fnr(logit)*p_coef
            case 'equalized odds':
                loss_per_attr = pret_fnr(logit)*p_coef + pret_fpr(logit)*n_coef
            case _:
                assert False, f'unrecognized fairness criteria'
        return torch.mean(loss_per_attr)

    # -------------------- losses (categorical) --------------------
    def categori_filter_logit(self, logit, batch_dim=0):
        # check the out_feature match the logit shape
        match (self.out_feature, batch_dim):
            case (11, 0) | (16, 0) | (18, 0):
                assert logit.shape[1] == self.out_feature, "model out feature mismatch"
            case (11, 1) | (16, 1) | (18, 1):
                assert logit.shape[2] == self.out_feature, "model out feature mismatch"
            case _:
                assert False, "check the model out feature"
        # divide the logit
        match (self.out_feature, batch_dim):
            case (11, 0):
                return logit[:,0:2], logit[:,2:11]
            case (16, 0):
                return logit[:,0:5], logit[:,5:7], logit[:,7:16]
            case (18, 0):
                return logit[:,0:7], logit[:,7:9], logit[:,9:18]
            case (11, 1):
                return logit[:,:,0:2], logit[:,:,2:11]
            case (16, 1):
                return logit[:,:,0:5], logit[:,:,5:7], logit[:,:,7:16]
            case (18, 1):
                return logit[:,:,0:7], logit[:,:,7:9], logit[:,:,9:18]
    
    def categori_to_prediction(self, logit, batch_dim=0):
        # encode in index
        _, pred = torch.max(logit, dim=batch_dim+1)
        return pred
    
    def categori_to_approx_prediction(self, logit, batch_dim=0):
        # encode in one hot 
        softmaxed = F.softmax(logit, dim=batch_dim+1)
        pred = 1./(1+torch.exp(-1e2*(softmaxed-0.5)))
        return pred

    def categori_direct_loss(self, logit, label, sens):
        if self.out_feature == 11:
            gender_logit, age_logit = self.categori_filter_logit(logit, batch_dim=0)
            gender_pred, age_pred = map(self.categori_to_approx_prediction, [gender_logit, age_logit], [0,0])
            gender_label, age_label = map(F.one_hot, [label[:,0], label[:,1]], [2,9])
            gender_result, age_result = torch.sum(gender_pred*gender_label, dim=1), torch.sum(age_pred*age_label, dim=1)
            result = torch.stack((gender_result, age_result), dim=1)
        else:
            race_logit, gender_logit, age_logit = self.categori_filter_logit(logit, batch_dim=0)
            race_pred, gender_pred, age_pred = map(self.categori_to_approx_prediction, [race_logit, gender_logit, age_logit], [0,0,0])
            if self.out_feature == 16:
                race_label, gender_label, age_label = map(F.one_hot, [label[:,0], label[:,1], label[:,2]], [5,2,9])
            elif self.out_feature == 18:
                race_label, gender_label, age_label = map(F.one_hot, [label[:,0], label[:,1], label[:,2]], [7,2,9])
            race_result, gender_result, age_result = torch.sum(race_pred*race_label, dim=1), torch.sum(gender_pred*gender_label, dim=1), torch.sum(age_pred*age_label, dim=1)
            result = torch.stack((race_result, gender_result, age_result), dim=1)
        group_1_result, group_2_result = regroup_tensor_categori(result, sens, regroup_dim=0)
        group_1_total, group_2_total = group_1_result.shape[0], group_2_result.shape[0]
        # fairness target on age
        if self.out_feature == 11:
            group_1_acc, group_2_acc = group_1_result[:,1].sum()/(group_1_total+1e-9), group_2_result[:,1].sum()/(group_2_total+1e-9)
        else:
            group_1_acc, group_2_acc = group_1_result[:,2].sum()/(group_1_total+1e-9), group_2_result[:,2].sum()/(group_2_total+1e-9)
        loss = torch.abs(group_1_acc-group_2_acc)
        return torch.mean(loss)

    def categori_masking_loss(self, logit, label, sens):
        if self.out_feature == 11:
            gender_logit, age_logit = self.categori_filter_logit(logit, batch_dim=0)
            gender_pred, age_pred = map(self.categori_to_approx_prediction, [gender_logit, age_logit], [0,0])
            gender_label, age_label = map(F.one_hot, [label[:,0], label[:,1]], [2,9])
            gender_result, age_result = torch.sum(gender_pred*gender_label, dim=1), torch.sum(age_pred*age_label, dim=1)
            result = torch.stack((gender_result, age_result), dim=1)
        else:
            race_logit, gender_logit, age_logit = self.categori_filter_logit(logit, batch_dim=0)
            race_pred, gender_pred, age_pred = map(self.categori_to_approx_prediction, [race_logit, gender_logit, age_logit], [0,0,0])
            if self.out_feature == 16:
                race_label, gender_label, age_label = map(F.one_hot, [label[:,0], label[:,1], label[:,2]], [5,2,9])
            elif self.out_feature == 18:
                race_label, gender_label, age_label = map(F.one_hot, [label[:,0], label[:,1], label[:,2]], [7,2,9])
            race_result, gender_result, age_result = torch.sum(race_pred*race_label, dim=1), torch.sum(gender_pred*gender_label, dim=1), torch.sum(age_pred*age_label, dim=1)
            result = torch.stack((race_result, gender_result, age_result), dim=1)
        group_1_result, group_2_result = regroup_tensor_categori(result, sens, regroup_dim=0)
        group_1_total, group_2_total = group_1_result.shape[0], group_2_result.shape[0]
        # fairness target on age
        if self.out_feature == 11:
            group_1_acc, group_2_acc = group_1_result[:,1].sum()/(group_1_total+1e-9), group_2_result[:,1].sum()/(group_2_total+1e-9)
            loss_CE_age = F.cross_entropy(age_logit, label[:,1], reduction='none')
        else:
            group_1_acc, group_2_acc = group_1_result[:,2].sum()/(group_1_total+1e-9), group_2_result[:,2].sum()/(group_2_total+1e-9)
            loss_CE_age = F.cross_entropy(age_logit, label[:,2], reduction='none')
        group_1_coef, group_2_coef = torch.where(group_1_acc > group_2_acc, -1 ,0), torch.where(group_2_acc > group_1_acc, -1 ,0)
        group_1_CE, group_2_CE = regroup_tensor_categori(loss_CE_age, sens, regroup_dim=0)
        loss = torch.cat((group_1_CE*group_1_coef, group_2_CE*group_2_coef), 0) # in shape (N)
        return torch.mean(loss)

    def categori_perturb_loss(self, logit, label, sens):
        def perturbed_pq(x, label=label):
            if self.out_feature == 11:
                gender_logit, age_logit = self.categori_filter_logit(x, batch_dim=1)
                gender_pred, age_pred = map(self.categori_to_prediction, [gender_logit, age_logit], [1,1])
                label_duped = label.repeat(x.shape[0], 1, 1)
                gender_label, age_label = label_duped[:,:,0], label_duped[:,:,1]
                gender_result, age_result = torch.eq(gender_pred, gender_label), torch.eq(age_pred, age_label)
                result = torch.stack((gender_result, age_result), dim=2)
            else:
                race_logit, gender_logit, age_logit = self.categori_filter_logit(x, batch_dim=1)
                race_pred, gender_pred, age_pred = map(self.categori_to_prediction, [race_logit, gender_logit, age_logit], [0,0,0])
                label_duped = label.repeat(x.shape[0], 1, 1)
                race_label, gender_label, age_label = label_duped[:,:,0], label_duped[:,:,1], label_duped[:,:,2]
                race_result, gender_result, age_result = torch.sum(torch.eq(race_pred, race_label), dim=1), torch.sum(torch.eq(gender_pred, gender_label), dim=1), torch.sum(torch.eq(age_pred, age_label), dim=1)
                result = torch.stack((race_result, gender_result, age_result), dim=2)
            group_1_result, group_2_result = regroup_tensor_categori(result, sens, regroup_dim=1)
            group_1_total, group_2_total = group_1_result.shape[1], group_2_result.shape[1]
            # fairness target on age
            if self.out_feature == 11:
                group_1_acc, group_2_acc = torch.sum(group_1_result[:,:,1], dim=1)/(group_1_total+1e-9), torch.sum(group_2_result[:,:,1], dim=1)/(group_2_total+1e-9)
            else:
                group_1_acc, group_2_acc = torch.sum(group_1_result[:,:,2], dim=1)/(group_1_total+1e-9), torch.sum(group_2_result[:,:,2], dim=1)/(group_2_total+1e-9)
            perturbed_loss = torch.abs(group_1_acc-group_2_acc)
            return perturbed_loss
        # Turns a function into a differentiable one via perturbations
        pret_pq = perturbed(perturbed_pq, 
                             num_samples=10000,
                             sigma=0.5,
                             noise='gumbel',
                             batched=False)
        # loss perturbed_loss
        loss = pret_pq(logit)
        return torch.mean(loss)

    def categori_accuracy_perturb_loss(self, logit, label, sens, coef):
        def perturb_acc(x, label=label):
            if self.out_feature == 11:
                gender_logit, age_logit = self.categori_filter_logit(x, batch_dim=1)
                gender_pred, age_pred = map(self.categori_to_prediction, [gender_logit, age_logit], [1,1])
                label_duped = label.repeat(x.shape[0], 1, 1)
                gender_label, age_label = label_duped[:,:,0], label_duped[:,:,1]
                gender_result, age_result = torch.eq(gender_pred, gender_label), torch.eq(age_pred, age_label)
                result = torch.stack((gender_result, age_result), dim=2)
            else:
                race_logit, gender_logit, age_logit = self.categori_filter_logit(x, batch_dim=1)
                race_pred, gender_pred, age_pred = map(self.categori_to_prediction, [race_logit, gender_logit, age_logit], [0,0,0])
                label_duped = label.repeat(x.shape[0], 1, 1)
                race_label, gender_label, age_label = label_duped[:,:,0], label_duped[:,:,1], label_duped[:,:,2]
                race_result, gender_result, age_result = torch.sum(torch.eq(race_pred, race_label), dim=1), torch.sum(torch.eq(gender_pred, gender_label), dim=1), torch.sum(torch.eq(age_pred, age_label), dim=1)
                result = torch.stack((race_result, gender_result, age_result), dim=2)
            group_1_result, group_2_result = regroup_tensor_categori(result, sens, regroup_dim=1)
            group_1_total, group_2_total = group_1_result.shape[1], group_2_result.shape[1]
            # accuracy of age
            if self.out_feature == 11:
                group_1_acc, group_2_acc = torch.sum(group_1_result[:,:,1], dim=1)/(group_1_total+1e-9), torch.sum(group_2_result[:,:,1], dim=1)/(group_2_total+1e-9)
            else:
                group_1_acc, group_2_acc = torch.sum(group_1_result[:,:,2], dim=1)/(group_1_total+1e-9), torch.sum(group_2_result[:,:,2], dim=1)/(group_2_total+1e-9)
            perturbed_loss = ((1 - group_1_acc) + (1 - group_2_acc))
            return perturbed_loss
        # Turns a function into a differentiable one via perturbations
        pret_acc = perturbed(perturb_acc, 
                             num_samples=10000,
                             sigma=0.5,
                             noise='gumbel',
                             batched=False)
        loss_per_attr = pret_acc(logit)*coef
        return torch.mean(loss_per_attr)

    def run(self, logit, label, sens, coef=None, n_coef=None):
        recovery_loss = 0
        if self.pred_type == 'binary':
            if len(coef) and sum(coef) > 0:
                match self.loss_type:
                    case 'direct' | 'masking' | 'perturb optim':
                        pred = torch.where(logit> 0.5, 1, 0)
                        FN_mask, FP_mask = torch.sub(1, pred)*label, pred*torch.sub(1, label)
                        BCEloss = F.binary_cross_entropy(logit, label, reduction='none')
                        FN_BCE, FP_BCE = torch.mean(BCEloss*FN_mask, dim=0), torch.mean(BCEloss*FP_mask, dim=0)
                        recovery_loss = torch.mean(coef*FN_BCE+n_coef*FP_BCE)
                    case 'full perturb optim':
                        recovery_loss = self.binary_accuracy_perturb_loss(logit, label, sens, coef, n_coef)
            match self.loss_type:
                case 'direct':
                    loss = self.binary_direct_loss(logit, label, sens)
                case 'masking':
                    loss = self.binary_masking_loss(logit, label, sens)
                case 'perturb optim' | 'full perturb optim':
                    loss = self.binary_perturb_loss(logit, label, sens)
                case _:
                    assert False, f'do not support such loss type'
        elif self.pred_type == 'categorical':
            if len(coef) and sum(coef) > 0:
                match self.loss_type:
                    case 'direct' | 'masking' | 'perturb optim':
                        if self.out_feature == 11:
                            _, age_logit = self.categori_filter_logit(logit, batch_dim=0)
                            loss_CE_age = F.cross_entropy(age_logit, label[:,1], reduction='none')
                        else:
                            _, _, age_logit = self.categori_filter_logit(logit, batch_dim=0)
                            loss_CE_age = F.cross_entropy(age_logit, label[:,2], reduction='none')
                        loss_CE_age_per_attr = torch.mean(loss_CE_age, dim=0)
                        recovery_loss = torch.mean(coef*loss_CE_age)
                    case 'full perturb optim':
                        recovery_loss = self.categori_accuracy_perturb_loss(logit, label, sens, coef)
            match self.loss_type:
                case 'direct':
                    loss = self.categori_direct_loss(logit, label, sens)
                case 'masking':
                    loss = self.categori_masking_loss(logit, label, sens)
                case 'perturb optim' | 'full perturb optim':
                    loss = self.categori_perturb_loss(logit, label, sens)
                case _:
                    assert False, f'do not support such loss type'
        return loss + recovery_loss