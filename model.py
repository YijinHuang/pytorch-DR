import torch
import torch.nn as nn
import torch.nn.functional as F


class o_ONet(nn.Module):
    def __init__(self, net_size, input_size, feature_dim):
        super(o_ONet, self).__init__()

        # require inputs width and height in each layer because of the using of untied biases.
        sizes = self.cal_sizes(net_size, input_size)

        # named layers
        self.conv = nn.Sequential()
        if net_size in ['small', 'medium', 'large']:
            # 1-11 layers
            small_conv = nn.Sequential(
                self.basic_conv2d(3, 32, sizes[0], sizes[0], kernel_size=5, stride=2, padding=2),
                self.basic_conv2d(32, 32, sizes[0], sizes[0], kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                self.basic_conv2d(32, 64, sizes[1], sizes[1], kernel_size=5, stride=2, padding=2),
                self.basic_conv2d(64, 64, sizes[1], sizes[1], kernel_size=3, stride=1, padding=1),
                self.basic_conv2d(64, 64, sizes[1], sizes[1], kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                self.basic_conv2d(64, 128, sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
                self.basic_conv2d(128, 128, sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
                self.basic_conv2d(128, 128, sizes[2], sizes[2], kernel_size=3, stride=1, padding=1),
            )
            self.conv.add_module('small_conv', small_conv)

        if net_size in ['medium', 'large']:
            # 12-15 layers
            medium_conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                self.basic_conv2d(128, 256, sizes[3], sizes[3], kernel_size=3, stride=1, padding=1),
                self.basic_conv2d(256, 256, sizes[3], sizes[3], kernel_size=3, stride=1, padding=1),
                self.basic_conv2d(256, 256, sizes[3], sizes[3], kernel_size=3, stride=1, padding=1),
            )
            self.conv.add_module('medium_conv', medium_conv)

        if net_size in ['large']:
            # 16-18 layers
            large_conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
                self.basic_conv2d(256, 512, sizes[4], sizes[4], kernel_size=3, stride=1, padding=1),
                self.basic_conv2d(512, 512, sizes[4], sizes[4], kernel_size=3, stride=1, padding=1),
            )
            self.conv.add_module('large_conv', large_conv)

        # RMSPooling layer
        self.conv.add_module('rmspool', RMSPool(3, 3))

        # regression part
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feature_dim, 1024),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.5),
            nn.Linear(512, 1024),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 1)
        )

        # initial parameters
        for m in self.modules():
            if isinstance(m, Conv2dUntiedBias) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1)
                nn.init.constant_(m.bias, 0.05)

    def basic_conv2d(self, in_channels, out_channels, height, width, kernel_size, stride, padding):
        return nn.Sequential(
            Conv2dUntiedBias(in_channels, out_channels, height, width, kernel_size, stride, padding),
            nn.LeakyReLU(negative_slope=0.01)
        )

    def forward(self, x):
        features = self.conv(x)
        # reshape to satisify maxpool1d input shape requirement
        features = features.view(features.size(0), 1, -1)
        predict = self.fc(features)
        predict = torch.squeeze(predict)
        return predict

    # load part of pretrained_model like o_O solution \
    # using multi-scale image to train model by setting type to part \
    # or load full weights by setting type to full.
    def load_weights(self, pretrained_model, exclude=[]):
        pretrained_dict = torch.load(pretrained_model).state_dict()
        model_dict = self.state_dict()

        # exclude
        for name in list(pretrained_dict.keys()):
            # using untied biases will make it unable to reload.
            if 'bias' in name:
                pretrained_dict.pop(name)
                continue
            for e in exclude:
                if e in name:
                    pretrained_dict.pop(name)
                    break

        # load weights
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        return pretrained_dict

    def layer_configs(self):
        model_dict = self.state_dict()
        return [(tensor, model_dict[tensor].size()) for tensor in model_dict]

    def cal_sizes(self, net_size, input_size):
        sizes = []
        if net_size in ['small', 'medium', 'large']:
            sizes.append(self._reduce_size(input_size, 5, 2, 2))
            after_maxpool = self._reduce_size(sizes[-1], 3, 0, 2)
            sizes.append(self._reduce_size(after_maxpool, 5, 2, 2))
            after_maxpool = self._reduce_size(sizes[-1], 3, 0, 2)
            sizes.append(self._reduce_size(after_maxpool, 3, 1, 1))
        if net_size in ['medium', 'large']:
            after_maxpool = self._reduce_size(sizes[-1], 3, 0, 2)
            sizes.append(self._reduce_size(after_maxpool, 3, 1, 1))
        if net_size in ['large']:
            after_maxpool = self._reduce_size(sizes[-1], 3, 0, 2)
            sizes.append(self._reduce_size(after_maxpool, 3, 1, 1))

        return sizes

    def _reduce_size(self, input_size, kernel_size, padding, stride):
        return (input_size + (2 * padding) - (kernel_size - 1) - 1) // stride + 1


class RMSPool(nn.Module):
    def __init__(self, kernel_size, stride):
        super(RMSPool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = torch.pow(x, 2)
        x = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        x = torch.sqrt(x)
        return x


class Conv2dUntiedBias(nn.Module):
    def __init__(self, in_channels, out_channels, height, width, kernel_size, stride=1, padding=0):
        super(Conv2dUntiedBias, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels, height, width))

    def forward(self, x):
        output = F.conv2d(x, self.weight, None, self.stride, self.padding)
        output += self.bias.unsqueeze(0).repeat(x.size(0), 1, 1, 1)
        return output
