import torch
import torch.nn as nn
import torch.nn.functional as F


class o_ONet(nn.Module):
    def __init__(self, net_size, feature_dim):
        super(o_ONet, self).__init__()

        # 1-11 layers
        small_conv = nn.Sequential(
            self.basic_conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            self.basic_conv2d(32, 32, kernel_size=4, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            self.basic_conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            self.basic_conv2d(64, 64, kernel_size=4, stride=1, padding=2),
            self.basic_conv2d(64, 64, kernel_size=4, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            self.basic_conv2d(64, 128, kernel_size=4, stride=1, padding=2),
            self.basic_conv2d(128, 128, kernel_size=4, stride=1, padding=1),
            self.basic_conv2d(128, 128, kernel_size=4, stride=1, padding=2),
        )

        # 12-15 layers
        medium_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            self.basic_conv2d(128, 256, kernel_size=4, stride=1, padding=2),
            self.basic_conv2d(256, 256, kernel_size=4, stride=1, padding=1),
            self.basic_conv2d(256, 256, kernel_size=4, stride=1, padding=2),
        )

        # 16-17 layers (without 18 layer for net B)
        large_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            self.basic_conv2d(256, 512, kernel_size=4, stride=1, padding=1),
        )

        # name layers
        self.conv = nn.Sequential()
        if net_size in ['small', 'medium', 'large']:
            self.conv.add_module('small_conv', small_conv)
        if net_size in ['medium', 'large']:
            self.conv.add_module('medium_conv', medium_conv)
        if net_size in ['large']:
            self.conv.add_module('large_conv', large_conv)
        self.conv.add_module('rmspool', RMSPool())

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
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def basic_conv2d(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
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
    def load_weights(self, pretrained_model, type='full'):
        pretrained_dict = torch.load(pretrained_model).state_dict()
        model_dict = self.state_dict()
        if type == 'part':
            pretrained_dict = {
                name: pretrained_tensor for name, pretrained_tensor in pretrained_dict.items()
                if name in model_dict and 'fc' not in name
            }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        return pretrained_dict

    def layer_configs(self):
        model_dict = self.state_dict()
        return [(tensor, model_dict[tensor].size()) for tensor in model_dict]


# o_ONet with batch normalization
class o_ONet2(nn.Module):
    def __init__(self):
        super(o_ONet2, self).__init__()

        self.conv = nn.Sequential(
            self.basic_conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            self.basic_conv2d(32, 32, kernel_size=4, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            self.basic_conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            self.basic_conv2d(64, 64, kernel_size=4, stride=1, padding=2),
            self.basic_conv2d(64, 64, kernel_size=4, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            self.basic_conv2d(64, 128, kernel_size=4, stride=1, padding=2),
            self.basic_conv2d(128, 128, kernel_size=4, stride=1, padding=1),
            self.basic_conv2d(128, 128, kernel_size=4, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            self.basic_conv2d(128, 256, kernel_size=4, stride=1, padding=2),
            self.basic_conv2d(256, 256, kernel_size=4, stride=1, padding=1),
            self.basic_conv2d(256, 256, kernel_size=4, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            self.basic_conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            RMSPool()
        )

        self.maxout = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01)
        self.dense_norm = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(512, 1024)
        self.regress = nn.Linear(512, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def basic_conv2d(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.01)
        )

    def forward(self, x):
        x = self.conv(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.dense_norm(self.fc1(x))
        x = x.view(x.size(0), 1, -1)
        x = self.maxout(x)
        x = self.lrelu(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.dense_norm(self.fc2(x))
        x = x.view(x.size(0), 1, -1)
        x = self.maxout(x)
        x = self.lrelu(x)

        x = self.regress(x)
        x = torch.squeeze(x)
        return x


class RMSPool(nn.Module):
    def forward(self, x):
        x = torch.pow(x, 2)
        x = F.avg_pool2d(x, kernel_size=3, stride=2)
        x = torch.sqrt(x)
        return x
