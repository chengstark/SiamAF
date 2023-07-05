import torch
import torch.nn as nn
import torch.nn.functional as F 
from resnet_zoo import *


class ResBlock(nn.Module):

    def __init__(self, filters, stage, i_block, kernel_size=3, stride=1):
        super(ResBlock,self).__init__()
        self.out_channels = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = filters

        if i_block == 0 and stage != 0:
            self.in_channels = filters // 2

        self.conv1 = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=1)
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=1)

        self.bn2 = nn.BatchNorm1d(self.out_channels)
        self.bn3 = nn.BatchNorm1d(self.out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):

        out = x
        out = self.relu1(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        if x.shape[1] != out.shape[1]:
            out = torch.add(out, torch.repeat_interleave(x, 2, dim=1))
        else:
            out = torch.add(out, x)

        out = self.relu2(self.bn3(out))

        return out



class Resnet34Baseline(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet34Baseline,self).__init__()
        self.num_classes = num_classes
        self.conv0 = nn.Conv1d(1, 48, 80, 4)
        self.bn0 = nn.BatchNorm1d(48)
        self.relu = nn.ReLU()
        self.pool0 = nn.MaxPool1d(4)
        self.stage0 = nn.Sequential(ResBlock(48, 0, 0),
                                   ResBlock(48, 0, 1))
        self.pool1 = nn.MaxPool1d(4)
        self.stage1 = nn.Sequential(ResBlock(96, 1, 0),
                                   ResBlock(96, 1, 1))
        self.pool2 = nn.MaxPool1d(4)
        self.stage2 = nn.Sequential(ResBlock(192, 2, 0),
                                    ResBlock(192, 2, 1))
        self.pool3 = nn.MaxPool1d(4)
        self.stage3 = nn.Sequential(ResBlock(384, 3, 0),
                                    ResBlock(384, 3, 1))

        self.avgpool = nn.AvgPool1d(1)
        self.dense1 = nn.Linear(384, self.num_classes)

    def forward(self, x, placeholder):
        
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)
        out = self.pool0(out)


        out = self.stage0(out)
        out = self.pool1(out)

        out = self.stage1(out)
        out = self.pool2(out)


        out = self.stage2(out)
        out = self.pool3(out)


        out = self.stage3(out)


        out = self.avgpool(out)
        features = out.mean(dim=2)

        out = self.dense1(features)

        return features, out


class Resnet34(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet34,self).__init__()
        self.num_classes = num_classes
        self.conv0 = nn.Conv1d(1, 48, 80, 4)
        self.bn0 = nn.BatchNorm1d(48)
        self.relu = nn.ReLU()
        self.pool0 = nn.MaxPool1d(4)
        self.stage0 = nn.Sequential(ResBlock(48, 0, 0),
                                   ResBlock(48, 0, 1))
        self.pool1 = nn.MaxPool1d(4)
        self.stage1 = nn.Sequential(ResBlock(96, 1, 0),
                                   ResBlock(96, 1, 1))
        self.pool2 = nn.MaxPool1d(4)
        self.stage2 = nn.Sequential(ResBlock(192, 2, 0),
                                    ResBlock(192, 2, 1))
        self.pool3 = nn.MaxPool1d(4)
        self.stage3 = nn.Sequential(ResBlock(384, 3, 0),
                                    ResBlock(384, 3, 1))

        self.avgpool = nn.AvgPool1d(1)
        self.fc = nn.Linear(384, self.num_classes)

    def forward(self, x):
        
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)
        out = self.pool0(out)


        out = self.stage0(out)
        out = self.pool1(out)

        out = self.stage1(out)
        out = self.pool2(out)


        out = self.stage2(out)
        out = self.pool3(out)


        out = self.stage3(out)


        out = self.avgpool(out)
        features = out.mean(dim=2)
        out = self.fc(features)

        return out, features


class DirectFwd(nn.Module):
    def __init__(self):
        super(DirectFwd, self).__init__()

    def forward(self, x):
        return x


class Res34SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, dim=2048, pred_dim=512, predictor=True, single_source_mode=False):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(Res34SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = Resnet34(num_classes=dim)
        self.single_source_mode = single_source_mode
        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        if predictor:
        # build a 2-layer predictor
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                            nn.BatchNorm1d(pred_dim),
                                            nn.ReLU(inplace=True), # hidden layer
                                            nn.Linear(pred_dim, dim)) # output layer
        else:
            self.predictor = DirectFwd()

        self.classification_head = nn.Linear(384, 2)

    def forward(self, ECG, PPG, ECG_classification_head, PPG_classification_head):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        if not self.single_source_mode:
            # compute features for one view
            z1, _ = self.encoder(ECG) # NxC
            z2, _ = self.encoder(PPG) # NxC

            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC

            _, features1 = self.encoder(ECG_classification_head)
            _, features2 = self.encoder(PPG_classification_head)
            
            class_pred1 = self.classification_head(features1)
            class_pred2 = self.classification_head(features2)

            return p1, p2, z1.detach(), z2.detach(), class_pred1, class_pred2
        else:
            z1, features1 = self.encoder(ECG) # NxC

            p1 = self.predictor(z1) # NxC
            
            class_pred1 = self.classification_head(features1)

            return p1, z1.detach(), class_pred1


class Res50SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, dim=2048, pred_dim=512, predictor=True, single_source_mode=False):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(Res50SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = ResNet50(num_classes=dim)
        self.single_source_mode = single_source_mode
        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]

        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        if predictor:
        # build a 2-layer predictor

            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                            nn.BatchNorm1d(pred_dim),
                                            nn.ReLU(inplace=True), # hidden layer
                                            nn.Linear(pred_dim, dim)) # output layer
        else:
            self.predictor = DirectFwd()

        self.classification_head = nn.Linear(1536, 2)

    def forward(self, ECG, PPG):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        if not self.single_source_mode:
            # compute features for one view
            z1, features1 = self.encoder(ECG) # NxC
            z2, features2 = self.encoder(PPG) # NxC

            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC
            
            class_pred1 = self.classification_head(features1)
            class_pred2 = self.classification_head(features2)

            return p1, p2, z1.detach(), z2.detach(), class_pred1, class_pred2
        else:
            z1, features1 = self.encoder(ECG) # NxC

            p1 = self.predictor(z1) # NxC
            
            class_pred1 = self.classification_head(features1)

            return p1, z1.detach(), class_pred1



class Resnet34Head(nn.Module):
    def __init__(self):
        super(Resnet34Head,self).__init__()
        self.conv0 = nn.Conv1d(1, 48, 80, 4)
        self.bn0 = nn.BatchNorm1d(48)
        self.relu = nn.ReLU()
        self.pool0 = nn.MaxPool1d(4)

    def forward(self, x):
        
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.relu(out)
        out = self.pool0(out)
        return out


class Resnet34Body(nn.Module):
    def __init__(self, num_classes=2):
        super(Resnet34Body,self).__init__()
        self.num_classes = num_classes
        self.stage0 = nn.Sequential(ResBlock(48, 0, 0),
                                   ResBlock(48, 0, 1))
        self.pool1 = nn.MaxPool1d(4)
        self.stage1 = nn.Sequential(ResBlock(96, 1, 0),
                                   ResBlock(96, 1, 1))
        self.pool2 = nn.MaxPool1d(4)
        self.stage2 = nn.Sequential(ResBlock(192, 2, 0),
                                    ResBlock(192, 2, 1))
        self.pool3 = nn.MaxPool1d(4)
        self.stage3 = nn.Sequential(ResBlock(384, 3, 0),
                                    ResBlock(384, 3, 1))

        self.avgpool = nn.AvgPool1d(1)
        self.fc = nn.Linear(384, self.num_classes)

    def forward(self, x):
        out = self.stage0(x)
        out = self.pool1(out)

        out = self.stage1(out)
        out = self.pool2(out)

        out = self.stage2(out)
        out = self.pool3(out)

        out = self.stage3(out)

        out = self.avgpool(out)
        features = out.mean(dim=2)
        out = self.fc(features)

        return out, features


class Res34SimSiamSplitHeads(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, dim=2048, pred_dim=512, predictor=True, single_source_mode=False, single_source_head=''):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(Res34SimSiamSplitHeads, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs

        self.single_source_mode = single_source_mode
        self.single_source_head = single_source_head

        if self.single_source_mode and self.single_source_head not in ['PPG', 'ECG']:
            raise Exception('single_source_head must be either ECG or PPG')

        self.ECG_head = Resnet34Head()
        self.PPG_head = Resnet34Head()
        self.encoder = Resnet34Body(num_classes=dim)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        if predictor:
        # build a 2-layer predictor
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                            nn.BatchNorm1d(pred_dim),
                                            nn.ReLU(inplace=True), # hidden layer
                                            nn.Linear(pred_dim, dim)) # output layer
        else:
            self.predictor = DirectFwd()

        self.classification_head = nn.Linear(384, 2)

    def forward(self, ECG, PPG):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        if not self.single_source_mode:
            ECG_aft_head = self.ECG_head(ECG)
            PPG_aft_head = self.PPG_head(PPG)

            # compute features for one view
            z1, features1 = self.encoder(ECG_aft_head) # NxC
            z2, features2 = self.encoder(PPG_aft_head) # NxC

            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC
            
            class_pred1 = self.classification_head(features1)
            class_pred2 = self.classification_head(features2)

            return p1, p2, z1.detach(), z2.detach(), class_pred1, class_pred2

        elif self.single_source_mode and self.single_source_head == 'PPG':
            PPG_aft_head = self.PPG_head(PPG)

            # compute features for one view
            z2, features2 = self.encoder(PPG_aft_head) # NxC

            p2 = self.predictor(z2) # NxC
            
            class_pred2 = self.classification_head(features2)

            return p2, z2.detach(), class_pred2

        elif self.single_source_mode and self.single_source_head == 'ECG':
            ECG_aft_head = self.ECG_head(ECG)

            # compute features for one view
            z1, features1 = self.encoder(ECG_aft_head) # NxC

            p1 = self.predictor(z1) # NxC
            
            class_pred1 = self.classification_head(features1)

            return p1, z1.detach(), class_pred1