import torch
import torch.nn as nn

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
            padding='same')
        self.bn1 = nn.BatchNorm1d(self.out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding='same')

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
        self.dense1 = nn.Linear(384, self.num_classes)

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

        out = self.dense1(features)

        return features, out


