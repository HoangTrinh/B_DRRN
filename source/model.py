import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable
import functools

class BDRRN(nn.Module):
    def __init__(self, n_channels=3):
        super(BDRRN, self).__init__()
        self.input = nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(n_channels)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x, p):
        #Main branch
        residual = x
        inputs = self.input(self.relu(self.bn(x)))
        out = inputs
        s_inputs = self.input(self.relu(self.bn(p)))
        s_out = s_inputs

        for i in range(9):
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, inputs)
            if i < 3:
                s_out = self.conv2(self.relu(self.conv1(self.relu(s_out))))
                s_out = torch.add(s_out, s_inputs)

        out = torch.add(out, s_out)
        out = self.conv1(self.relu(out))
        out = self.conv2(self.relu(out))
        out = self.output(self.relu(out))
        out = torch.add(out, residual)
        return out

class BDRRN_cat(nn.Module):
    def __init__(self, n_channels=3):
        super(BDRRN_cat, self).__init__()
        self.input = nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.convcat = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=n_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(n_channels)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x, p):
        #Main branch
        residual = x
        inputs = self.input(self.relu(self.bn(x)))
        out = inputs
        #Sub branch
        s_inputs = self.input(self.relu(self.bn(p)))
        s_out = s_inputs

        for i in range(9):
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, inputs)
            if i < 3:
                s_out = self.conv2(self.relu(self.conv1(self.relu(s_out))))
                s_out = torch.add(s_out, s_inputs)

        out = torch.cat((out, s_out),dim =1)
        out = self.convcat(self.relu(out))
        out = self.conv1(self.relu(out))
        out = self.output(self.relu(out))
        out = torch.add(out, residual)
        return out
