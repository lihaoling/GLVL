
import os
import torch
import logging
import torchvision
from torch import nn
from os.path import join
from transformers import ViTModel


from model.cct import cct_14_7x2_384
from model.aggregation import Flatten
from model.normalization import L2Norm
import model.aggregation as aggregation


class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.arch_name = args.backbone

        self.aggregation = aggregation.GeM(work_with_tokens=False)
        self.aggregation = nn.Sequential(L2Norm(), self.aggregation, Flatten())

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


def get_aggregation(args):

    return aggregation.GeM(work_with_tokens=args.work_with_tokens)

def get_backbone(args):
    # The aggregation layer works differently based on the type of architecture

    if args.backbone.startswith("resnet18"):
        backbone = torchvision.models.resnet18(weights=True)
    elif args.backbone.startswith("resnet50"):
        backbone = torchvision.models.resnet50(weights=True)
    elif args.backbone.startswith("resnet101"):
        backbone = torchvision.models.resnet101(weights=True)

    ## 3 channels -> 1 channel
    backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    for name, child in backbone.named_children():
        # Freeze layers before conv_3
        if name == "layer3":
            break
        for params in child.parameters():
            params.requires_grad = False
    if args.backbone.endswith("conv4"):
        logging.debug(f"Train only conv4_x of the resnet{args.backbone.split('conv')[0]} (remove conv5_x), freeze the previous ones")
        layers = list(backbone.children())[:-3]
    elif args.backbone.endswith("conv5"):
        logging.debug(f"Train only conv4_x and conv5_x of the resnet{args.backbone.split('conv')[0]}, freeze the previous ones")
        layers = list(backbone.children())[:-2]

    backbone = torch.nn.Sequential(*layers)
    args.features_dim = get_output_channels_dim(backbone)  # Dinamically obtain number of channels in output
    return backbone

def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 1, 224, 224])).shape[1]

