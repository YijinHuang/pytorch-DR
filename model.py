import torch
import torch.nn as nn
import torch.nn.functional as F


class o_ONet(nn.Module):
    def __init__(self, net_size, input_size, feature_dim):
        super(o_ONet, self).__init__()

        self.small_conv = nn.Sequential(
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

        self.medium_conv = nn.Sequential(
            self.small_conv,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            self.basic_conv2d(128, 256, kernel_size=4, stride=1, padding=2),
            self.basic_conv2d(256, 256, kernel_size=4, stride=1, padding=1),
            self.basic_conv2d(256, 256, kernel_size=4, stride=1, padding=2),
        )

        self.large_conv = nn.Sequential(
            self.medium_conv,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            self.basic_conv2d(256, 512, kernel_size=4, stride=1, padding=1),
        )

        if net_size == 'small':
            self.conv = self.small_conv
        elif net_size == 'medium':
            self.conv = self.medium_conv
        elif net_size == 'large':
            self.conv = self.large_conv

        self.conv = nn.Sequential(
            self.conv,
            RMSPool()
        )

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
        features = features.view(features.size(0), 1, -1)
        predict = self.fc(features)
        predict = torch.squeeze(predict)
        return predict


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
