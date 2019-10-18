from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
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



class ConvBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self._layer = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                              stride=stride, padding=padding, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        return self._layer(x)


class CSAttention(nn.Module):
    """
    Pyramid Attention Network for Semantic Segmentation
    (https://arxiv.org/pdf/1805.10180.pdf)

    Selective Feature Connection Mechanism: Concatenating
    Multi-layer CNN Features with a Feature Selector
    (https://arxiv.org/pdf/1811.06295v1.pdf)
    """

    def __init__(self, a_channels, b_channels):
        super().__init__()

        self._channel_conv = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                           ConvBatchNormRelu(b_channels, b_channels, kernel_size=1),
                                           nn.Conv2d(b_channels, a_channels, kernel_size=1))
        self._spatial_conv = nn.Sequential(ConvBatchNormRelu(b_channels, b_channels, kernel_size=3, padding=1),
                                           nn.Conv2d(b_channels, 1, kernel_size=3, padding=1))

    def forward(self, a, b):
        ca = self._channel_conv(b)
        sa = self._spatial_conv(b)
        out = a * torch.sigmoid(ca * sa)

        return out


class DoubleAttention(nn.Module):
    """
    A2-Nets: Double Attention Networks
    (https://arxiv.org/pdf/1810.11579.pdf)
    """

    def __init__(self, a_channels, b_channels, n_attn_maps=512):
        super().__init__()

        self._attn_maps_conv = nn.Sequential(nn.Conv2d(b_channels, n_attn_maps, kernel_size=3, padding=1),
                                             nn.Softmax2d())
        self._attn_vectors_conv = nn.Sequential(nn.Conv2d(a_channels, n_attn_maps, kernel_size=3, padding=1),
                                                nn.Softmax(dim=1))

    def forward(self, a, b):
        attn_maps = self._attn_maps_conv(b)
        attn_maps = attn_maps.view(*attn_maps.shape[:2], -1)
        attn_maps = torch.transpose(attn_maps, 1, 2)

        attn_vectors = self._attn_vectors_conv(a)
        attn_vectors = attn_vectors.view(*attn_vectors.shape[:2], -1)

        descriptors = torch.bmm(a.view(*a.shape[:2], -1), attn_maps)
        z = torch.bmm(descriptors, attn_vectors)
        out = a + z.view_as(a)

        return out


class SpatialPyramidPolling(nn.Module):
    TYPES = ('avg', 'max')

    def __init__(self, sizes=((4, 4), (2, 2), (1, 1)), pool_type='max'):
        super().__init__()

        if pool_type not in SpatialPyramidPolling.TYPES:
            raise ValueError(f'Incorrect pooling type: expected types {SpatialPyramidPolling.TYPES}, got {pool_type}')

        self._pools = nn.ModuleList(
            [nn.AdaptiveMaxPool2d(size) if pool_type == 'max' else nn.AdaptiveAvgPool2d(size) for size in sizes]
        )

    def forward(self, x):
        batch_size = x.shape[0]
        pool_outs = [p(x).view(batch_size, -1) for p in self._pools]
        out = torch.cat(pool_outs, dim=-1)

        return out


class Attention(nn.Module):
    TYPES = ('none', 'cs_attention', 'double_attention')

    def __init__(self, attention_type, l_channels, h_channels):
        super().__init__()

        if attention_type not in Attention.TYPES:
            raise ValueError(f'Incorrect attention type: expected types {Attention.TYPES}, got {attention_type}')

        if attention_type == 'none':
            self._attn = lambda x, _: x
        elif attention_type == 'cs_attention':
            self._attn = CSAttention(l_channels, h_channels)
        elif attention_type == 'double_attention':
            self._attn = DoubleAttention(l_channels, h_channels)

    def forward(self, a, b):
        return self._attn(a, b)


class Upsampler(nn.Module):
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self._layer = nn.Sequential(nn.Upsample(scale_factor=scale_factor),
                                    ConvBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding))

    def forward(self, x):
        return self._layer(x)


class Linker(nn.Module):
    def __init__(self, in_channels, out_channels, attention_type='none', attention_post=False):
        super().__init__()

        self._conv = ConvBatchNormRelu(2 * in_channels, out_channels, kernel_size=3, padding=1)

        self._attention_post = attention_post
        attn_channels = out_channels if attention_post else in_channels
        self._attn = Attention(attention_type, attn_channels, attn_channels)

    def forward(self, x1, x2):
        assert x1.shape[1] == x2.shape[1]

        if self._attention_post:
            x = torch.cat([x1, x2], dim=1)
            x = self._conv(x)
            x = self._attn(x, x)

        else:
            x1 = self._attn(x1, x2)
            x = torch.cat([x1, x2], dim=1)
            x = self._conv(x)

        return x


class FPN(nn.Module):
    def __init__(self, enc_channels, fpn_channels, reduction, attention_type='none', attention_post=False):
        super().__init__()

        self._in_comprs = nn.ModuleList([ConvBatchNormRelu(c, fpn_channels, kernel_size=3, padding=1)
                                         for c in enc_channels])
        #self._l5_compr = ConvBatchNormRelu(in_channels, fpn_channels, kernel_size=1)
        #self._avgpool = nn.AdaptiveAvgPool2d(1)

        self._upsamplers = nn.ModuleList([Upsampler(2, fpn_channels, fpn_channels, kernel_size=3, padding=1)
                                          for _ in range(len(enc_channels)-1)])
        self._linkers = nn.ModuleList([Linker(fpn_channels, fpn_channels, attention_type, attention_post)
                                       for _ in range(len(enc_channels)-1)])
        self._out_comprs = nn.ModuleList([ConvBatchNormRelu(fpn_channels, fpn_channels // reduction, kernel_size=3, padding=1)
                                          for _ in range(len(enc_channels))])

    def forward(self, l0, l1, l2, l3, l4):
        #l5 = self._l5_compr(self._avgpool(l4))
        l0, l1, l2, l3, l4 = (compr(l) for compr, l in zip(self._in_comprs, [l0, l1, l2, l3, l4]))

        p4 = l4 #+ l5

        up4 = self._upsamplers[3](p4)
        p3 = self._linkers[3](l3, up4)

        up3 = self._upsamplers[2](p3)
        p2 = self._linkers[2](l2, up3)

        up2 = self._upsamplers[1](p2)
        p1 = self._linkers[1](l1, up2)

        up1 = self._upsamplers[0](p1)
        p0 = self._linkers[0](l0, up1)

        p0, p1, p2, p3, p4 = (compr(l) for compr, l in zip(self._out_comprs, [p0, p1, p2, p3, p4]))

        return p0, p1, p2, p3, p4


class ResFPN(nn.Module):
    def __init__(self, in_channels, out_channels, model='resnet34', pretrained=True, fpn_channels=256,
                 reduction=4, dropout=0, attention_type='none', attention_post=False):
        super().__init__()

        self._encoder = Encoder(in_channels, model, pretrained)
        self._fpn = FPN(self._encoder.feature_sizes, fpn_channels, reduction, attention_type, attention_post)

        self._upsampler4 = nn.Upsample(scale_factor=16)
        self._upsampler3 = nn.Upsample(scale_factor=8)
        self._upsampler2 = nn.Upsample(scale_factor=4)
        self._upsampler1 = nn.Upsample(scale_factor=2)

        self._out_convs = nn.Sequential(ConvBatchNormRelu(5 * (fpn_channels // reduction), fpn_channels, kernel_size=3, padding=1),
                                        nn.Dropout2d(dropout),
                                        nn.Conv2d(fpn_channels, out_channels, kernel_size=3, padding=1))

        self.set_activation(nn.CELU(inplace=True))
        self.set_gn()

    def forward(self, x, last_interp=True):
        l0, l1, l2, l3, l4 = self._encoder(x)
        p0, p1, p2, p3, p4 = self._fpn(l0, l1, l2, l3, l4)

        p1 = self._upsampler1(p1)
        p2 = self._upsampler2(p2)
        p3 = self._upsampler3(p3)
        p4 = self._upsampler4(p4)

        p = torch.cat([p0, p1, p2, p3, p4], dim=1)

        out = self._out_convs(p)

        if last_interp:
            out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        return out

    def set_gn(self, n_groups=16):
        def replace_bn(model):
            for child_name, child in model.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    setattr(model, child_name, nn.GroupNorm(n_groups, child.num_features))
                else:
                    replace_bn(child)

        replace_bn(self)

    def set_activation(self, activation):
        def replace_activation(model):
            for child_name, child in model.named_children():
                if isinstance(child, nn.ReLU):
                    setattr(model, child_name, activation)
                else:
                    replace_activation(child)

        replace_activation(self)