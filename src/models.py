from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

class BinaryModel(torch.nn.Module):
    """
    binary attribute prediction model,
    change the "out_feature" for number of attributes predicted.
        47: All attributes of MAADFace-HQ
    """

    def __init__(self, out_feature=46, weights='ResNet34_Weights.DEFAULT'):
        super(BinaryModel, self).__init__()
        self.model = models.resnet34(weights=weights)
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

class CategoricalModel(torch.nn.Module):
    """
    categorical attribute prediction model,
    change the "out_feature" for different model.
        16: UTKFace
        18: FairFace
    """

    def __init__(self, out_feature=18, weights='ResNet34_Weights.DEFAULT'):
        super(CategoricalModel, self).__init__()
        self.model = models.resnet34(weights=weights)
        in_feature = self.model.fc.in_features
        self.model.fc = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(in_feature, out_feature)),
        ]))
    
    def forward(self, x):
        """
        Input:
            x: Image of faces        (N, C, H, W)
        Output:
            z: attribute predictions (N, 16)
        """
        z = self.model(x)
        return z