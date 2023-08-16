import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WideResNet40(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet40, self).__init__()
        input_img = 32 # input image size: 32 for cifar10 and 256 for retinopathy and places
        self.kernel_size = input_img // 4 # avg pooling kernel size
        depth = 40 
        n = (depth - 4) // 6
        widen_factor = 2 # widen factor of the network
        self.dims_out = 64 * widen_factor # out_channels[3]
        out_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        drop_rate = 0.3 # drop rate of the network
        block = BasicBlock

        # layers
        self.conv1 = nn.Conv2d(3, out_channels[0], kernel_size=3, stride=1, padding=1, bias=False) # 1st conv before any network block
        self.block1 = NetworkBlock(n, out_channels[0], out_channels[1], block, 1, drop_rate) # 1st block
        self.block2 = NetworkBlock(n, out_channels[1], out_channels[2], block, 2, drop_rate) # 2nd block
        self.block3 = NetworkBlock(n, out_channels[2], out_channels[3], block, 2, drop_rate) # 3rd block
        self.bn1 = nn.BatchNorm2d(out_channels[3]) 
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(out_channels[3], num_classes)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)

        features = out.view(-1, self.dims_out)
        logits = self.fc(features)

        return logits, features
    
    def intermediate_forward(self, x, layer_index):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out
    
    def feature_list(self, x):
        out_list = [] 
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out_list.append(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.dims_out)
        return self.fc(out), out_list
         


class BasicBlock(nn.Module):
    def __init__(self, input_plane, output_plane, stride, droprate):
        self.input_plane = input_plane
        self.output_plane = output_plane
        self.stride = stride
        self.droprate = droprate

        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(self.input_plane)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.input_plane, self.output_plane, kernel_size=3, stride=self.stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.output_plane)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.output_plane, self.output_plane, kernel_size=3, stride=1, padding=1, bias=False)
        self.equalInOut = (self.input_plane == self.output_plane)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(self.input_plane, self.output_plane, kernel_size=1, stride=self.stride, padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)



class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

