############
#train_joint
############

import os
import torch
import logging
import torchvision
from torch import nn

from model.aggregation import Flatten
from model.normalization import L2Norm
import model.aggregation as aggregation


class GeoLocalizationNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        self.arch_name = args.backbone

        # Encoder
        self.backbone = torchvision.models.resnet50(weights=True)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.backbone.fc = nn.Linear(in_features=2048, out_features=128, bias=True)

        layers = nn.ModuleList([self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool,
                                self.backbone.layer1, self.backbone.layer2])


        for params in layers.parameters():
            params.requires_grad = False

        self.encoder = torch.nn.Sequential(*layers)



        # retrievalNet
        self.retrievalNet = retrievalNet(args)



        # SuperPointNet
        self.SuperPointNet = SuperPointNet(args)


    def forward(self, x, flag):
        x = self.encoder(x)


        if flag == 'retrieval':
            x1 = self.retrievalNet(x)
        else:
            x1 = self.SuperPointNet(x)

        return x1



class retrievalNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = torchvision.models.resnet50(weights=True)
        layers = nn.ModuleList([self.backbone.layer3, self.backbone.layer4, self.backbone.avgpool, self.backbone.fc])
        self.layers = torch.nn.Sequential(*layers)

        self.arch_name = args.backbone

        self.aggregation = aggregation.GeM(work_with_tokens=False)
        self.aggregation = nn.Sequential(L2Norm(), self.aggregation, Flatten())

        # args.features_dim = get_output_channels_dim(args, self.backbone)
        args.features_dim = 2048

    def forward(self, x):
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        # x = self.backbone.avgpool(x)
        # x = self.backbone.fc(x)

        x = self.aggregation(x)
        return x


class SuperPointNet(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, args):
        super().__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65

        self.relu = torch.nn.ReLU(inplace=True)

        # Detector Head.
        self.convPa = torch.nn.Conv2d(512, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(512, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)
        self.output = None


    def forward(self, x):



        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x)))
        semi = self.bnPb(self.convPb(cPa))
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x)))
        desc = self.bnDb(self.convDb(cDa))

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        output = {'semi': semi, 'desc': desc}
        self.output = output

        return output






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

def get_output_channels_dim(args, model):
    """Return the number of channels in the output of a model."""

    # return model(torch.ones([1, 3, args.resize[0], args.resize[1]])).shape[1]

