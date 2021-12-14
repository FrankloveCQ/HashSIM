import torch
import torch.nn as nn
import torchvision.models as models
import os
from torch.nn import Parameter
import math
import torch.nn.functional as F 
def load_model(arch, code_length,  num_cluster=30):
    """
    Load CNN model.

    Args
        arch(str): Model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    if arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier = model.classifier[:-2]
        model = ModelWrapper(model, 4096, code_length,num_cluster)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier = model.classifier[:-3]
        model = ModelWrapper(model, 4096, code_length,num_cluster)
    else:
        raise ValueError("Invalid model name!")

    return model


class ModelWrapper(nn.Module):
    """
    Add tanh activate function into model.

    Args
        model(torch.nn.Module): CNN model.
        last_node(int): Last layer outputs size.
        code_length(int): Hash code length.
    """
    def __init__(self, model, last_node, code_length, num_cluster):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.code_length = code_length
        self.hash_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(last_node, code_length),
            nn.Tanh(),
        )
        # self.cluster_layer = Parameter(torch.Tensor(num_cluster, code_length))
        #self.hash_targets = torch.nn.Parameter(hash_targets,requires_grad=True)
        # torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # Extract features
        self.extract_features = False

    def forward(self, x):
        if self.extract_features:
            return self.model(x)
        else:
            feature = self.model(x)
            y = self.hash_layer(feature)
            return y

    def set_extract_features(self, flag):
        """
        Extract features.

        Args
            flag(bool): true, if one needs extract features.
        """
        self.extract_features = flag


