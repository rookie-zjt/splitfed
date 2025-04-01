import math

from torch import nn
import torch.nn.functional as F
from config import spilt

# Model at server side
class Baseblock(nn.Module):
    expansion = 1

    def __init__(self, input_planes, planes, stride=1, dim_change=None):
        super(Baseblock, self).__init__()
        self.conv1 = nn.Conv2d(input_planes, planes, stride=stride, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, stride=1, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dim_change = dim_change

    def forward(self, x):
        res = x
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))

        if self.dim_change is not None:
            res = self.dim_change(res)

        output += res
        output = F.relu(output)

        return output


class ResNet18_server_side(nn.Module):
    def __init__(self, num_layers, classes, block = Baseblock ):
        super(ResNet18_server_side, self).__init__()
        self.input_planes = 64
        if spilt < 1:
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        if spilt < 2:
            self.layer2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
            )
        if spilt < 3:
            self.layer3 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
            )
        if spilt < 4:
            self.layer4 = self._layer(block, 128, num_layers[0], stride=2)
        if spilt < 5:
            self.layer5 = self._layer(block, 256, num_layers[1], stride=2)
        if spilt < 6:
            self.layer6 = self._layer(block, 512, num_layers[2], stride=2)

            self.averagePool = nn.AvgPool2d(kernel_size=7, stride=1)

            self.fc = nn.Linear(512 * block.expansion, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _layer(self, block, planes, num_layers, stride=2):
        dim_change = None
        if stride != 1 or planes != self.input_planes * block.expansion:
            dim_change = nn.Sequential(
                nn.Conv2d(self.input_planes, planes * block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes * block.expansion))
        netLayers = []
        netLayers.append(block(self.input_planes, planes, stride=stride, dim_change=dim_change))
        self.input_planes = planes * block.expansion
        for i in range(1, num_layers):
            netLayers.append(block(self.input_planes, planes))
            self.input_planes = planes * block.expansion

        return nn.Sequential(*netLayers)

    def forward(self, x):
        x1 = x
        x2 = x
        x3 = x
        x4 = x
        x5 = x
        y_hat = x

        if spilt < 1:
            x1 = F.relu(self.layer1(x))
        if spilt < 2:
            out1 = self.layer2(x1)
            out1 = out1 + x1  # adding the resudial inputs -- downsampling not required in this layer
            x2 = F.relu(out1)

        if spilt < 3:
            out2 = self.layer3(x2)
            out2 = out2 + x  # adding the resudial inputs -- downsampling not required in this layer
            x3 = F.relu(out2)

        if spilt < 4:
            x4 = self.layer4(x3)

        if spilt < 5:
            x5 = self.layer5(x4)

        if spilt < 6:
            x6 = self.layer6(x5)

            # x7 = F.avg_pool2d(x6, 7)
            x7 = F.avg_pool2d(x6, 2)  # 7*7卷积核太大了
            x8 = x7.view(x7.size(0), -1)
            y_hat = self.fc(x8)

        return y_hat