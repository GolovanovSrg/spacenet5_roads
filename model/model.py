from collections import OrderedDict

import torch
import torch.nn as nn
from pretrainedmodels import resnet34, se_resnext50_32x4d, se_resnext101_32x4d


class Encoder(nn.Module):
    def __init__(self, in_channels, model='resnet34', pretrained=True):
        super().__init__()

        assert model in ['resnet34', 'seresnext50', 'seresnext101']

        pretrained_dataset = 'imagenet' if pretrained else None
        if model == 'resnet34':
            self.model = resnet34(pretrained=pretrained_dataset)
            self.feature_sizes = [64, 64, 128, 256, 512]
        elif model == 'seresnext50':
            self.model = se_resnext50_32x4d(pretrained=pretrained_dataset)
            self.feature_sizes = [64, 64 * 4, 128 * 4, 256 * 4, 512 * 4]
        elif model == 'seresnext101':
            self.model = se_resnext101_32x4d(pretrained=pretrained_dataset)
            self.feature_sizes = [64, 64 * 4, 128 * 4, 256 * 4, 512 * 4]
        else:
            assert False


        layer0_modules = [
            ('conv1', nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),

            ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),

            ('conv3', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(64)),
            ('relu3', nn.ReLU(inplace=True)),
        ]
        self.model.max_pool = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.model.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.model.last_linear = None
        if hasattr(self.model, 'avg_pool'):
            self.model.avg_pool = None
        elif hasattr(self.model, 'avgpool'):
            self.model.avgpool = None
        else:
            assert False

    def forward(self, x):
        x0 = self.model.layer0(x)
        pooled_x0 = self.model.max_pool(x0)
        x1 = self.model.layer1(pooled_x0)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)

        return x0, x1, x2, x3, x4


class DecoderLinkerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        layer_modules = [
            ('conv', nn.Conv2d(2 * in_channels, out_channels, 3, padding=1)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]
        self.layer = nn.Sequential(OrderedDict(layer_modules))

    def forward(self, dec_x, enc_x):
        assert dec_x.shape[1] == enc_x.shape[1]
        x = torch.cat([dec_x, enc_x], dim=1)
        return self.layer(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        layer_modules = [('upsample', nn.Upsample(scale_factor=2)),
                         ('conv', nn.Conv2d(in_channels, out_channels, 3, padding=1)),
                         ('bn', nn.BatchNorm2d(out_channels)),
                         ('relu', nn.ReLU(inplace=True))
        ]
        self.layer = nn.Sequential(OrderedDict(layer_modules))

    def forward(self, x):
        return self.layer(x)


class Decoder(nn.Module):
    def __init__(self, enc_channels, out_channels):
        super().__init__()

        self.linker_layers = nn.ModuleList([DecoderLinkerBlock(f, f)
                                            for f in reversed(enc_channels[:-1])])
        self.layers = nn.ModuleList([DecoderBlock(enc_channels[i], enc_channels[i-1])
                                     for i in reversed(range(1, len(enc_channels)))])

        self.last_upsample = DecoderBlock(enc_channels[0], enc_channels[0] // 2)
        self.final = nn.Conv2d(enc_channels[0] // 2, out_channels, 3, padding=1)

    def forward(self, x0, x1, x2, x3, x4):
        enc_features = [x3, x2, x1, x0]
        x = x4
        for enc_feature, layer, linker_layer in zip(enc_features, self.layers, self.linker_layers):
            x = layer(x)
            x = linker_layer(x, enc_feature)

        x = self.last_upsample(x)
        out = self.final(x)

        return out


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, model='resnet34', pretrained=True):
        super().__init__()

        self.encoder = Encoder(in_channels, model, pretrained)
        self.decoder = Decoder(self.encoder.feature_sizes, out_channels)

    def forward(self, x):
        x0, x1, x2, x3, x4 = self.encoder(x)
        out = self.decoder(x0, x1, x2, x3, x4)

        return out

    def set_activation(self, activation):
        def replace_activation(model):
            for child_name, child in model.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(model, child_name, activation)
                else:
                    replace_activation(child)

        replace_activation(self)
