import torch
import torch.nn as nn


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, deploy=True):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy

        if deploy:
            self.reparam_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        else:
            self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn3x3 = nn.BatchNorm2d(out_channels)
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            self.bn1x1 = nn.BatchNorm2d(out_channels)
            self.bn_identity = nn.BatchNorm2d(in_channels) if in_channels == out_channels and stride == 1 else None

        self.nonlinearity = nn.ReLU()

    def forward(self, x):
        if self.deploy:
            return self.nonlinearity(self.reparam_conv(x))

        out = self.conv3x3(x)
        out = self.bn3x3(out)
        if self.bn_identity is not None:
            identity = self.bn_identity(x)
        else:
            identity = 0
        out += self.conv1x1(x) + self.bn1x1(self.conv1x1(x)) + identity
        return self.nonlinearity(out)

    def switch_to_deploy(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam_conv = nn.Conv2d(self.conv3x3.in_channels, self.conv3x3.out_channels, kernel_size=3,
                                      stride=self.conv3x3.stride, padding=1, bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        self.__delattr__('conv3x3')
        self.__delattr__('bn3x3')
        self.__delattr__('conv1x1')
        self.__delattr__('bn1x1')
        if self.bn_identity is not None:
            self.__delattr__('bn_identity')
        self.deploy = True

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_conv_bn(self.conv3x3, self.bn3x3)
        kernel1x1, bias1x1 = self._fuse_conv_bn(self.conv1x1, self.bn1x1)
        kernel_identity, bias_identity = self._fuse_conv_bn_identity()
        kernel = kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernel_identity
        bias = bias3x3 + bias1x1 + bias_identity
        return kernel, bias

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_conv_bn(self, conv, bn):
        if conv is None:
            return 0, 0
        else:
            kernel = conv.weight
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps

            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std

    def _fuse_conv_bn_identity(self):
        if self.bn_identity is None:
            return 0, 0
        else:
            input_dim = self.conv3x3.in_channels
            kernel = torch.zeros((input_dim, input_dim, 3, 3), dtype=self.conv3x3.weight.dtype,
                                 device=self.conv3x3.weight.device)
            for i in range(input_dim):
                kernel[i, i, 1, 1] = 1
            running_mean = self.bn_identity.running_mean
            running_var = self.bn_identity.running_var
            gamma = self.bn_identity.weight
            beta = self.bn_identity.bias
            eps = self.bn_identity.eps

            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std


class LightWeightNetwork(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, block=RepVGGBlock, num_blocks=[2, 2, 2, 2],
                 nb_filter=[4, 8, 16, 32, 64]):
        super(LightWeightNetwork, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        self.conv1_0 = self._make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_0 = self._make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_0 = self._make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])
        self.conv4_0 = self._make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])

        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


