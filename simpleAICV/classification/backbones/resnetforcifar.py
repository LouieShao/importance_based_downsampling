'''
Deep Residual Learning for Image Recognition
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
'''
import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simpleAICV.classification.common import load_state_dict
from tools.path import pretrained_models_path

import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import conv1x1, BasicBlock, Bottleneck

__all__ = [
    'ResNetCifar',
    'resnet18cifar',
    'resnet34cifar',
    'resnet50cifar',
    'resnet101cifar',
    'resnet152cifar',
]

model_urls = {
    'resnet18cifar': 'empty',
    'resnet34cifar': 'empty',
    'resnet50cifar': 'empty',
    'resnet101cifar': 'empty',
    'resnet152cifar': 'empty',
}

class AdjustableResidual(nn.Module):
    def __init__(self,):
        super(AdjustableResidual, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1))
        
        # nn.Parameter is a special kind of Variable, that will get
        # automatically registered as Module's parameter once it's assigned
        # 这个很重要！ Parameters是默认需要梯度的！
        
        # Not a very smart way to initialize weights
        self.weight.data.zero_()
        
    def forward(self, input1,input2):
        
        input = input1 * (self.weight+1) + input2 * (1 - self.weight)
        
        
class convconv(nn.Module):
    def __init__(self,):
        super(convconv, self).__init__()
        self.conv = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride=2, padding=1, groups = 32)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride=2, padding=1, groups = 32)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride=2, padding=1, groups = 32)
        self.conv4 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 2, stride=2, padding=1, groups = 32)
        self.unfold = torch.nn.Unfold(kernel_size=(3, 3), stride=1)
        
    def forward(self, a):
        batch_restore = a.size(0)
        b = self.conv(a)
        b = torch.relu(b)
        c = self.conv2(b)
        c = torch.relu(c)
        d = self.conv3(c)
        d = torch.relu(d)
        e = self.conv4(d)
        e = torch.sigmoid(e)
        a_ = a.unsqueeze(2)
        a_ = a_.view(batch_restore*64,1,32,32)
        e_ = e.view(batch_restore,64,9,1)
        a_ = F.pad(a_, (1,1,1,1), mode='reflect')
        a__ = self.unfold(a_)
        a___ = a__.view(batch_restore,64,9,1024)
        a___ = a___.transpose(2,3)
        xx = torch.matmul(a___,e_)
        xx = xx.view(batch_restore,64,32,32)
        return xx    
        
        
        
        
        
        
        return input
class norm_conv(nn.Module):
    def __init__(self,):
        super(norm_conv, self).__init__()
        self.unfold = torch.nn.Unfold(kernel_size=(3, 3), stride=1)
        self.fold = torch.nn.Fold(output_size=(32, 32), kernel_size=(1,1), stride=1)
        self.conv_kernel = nn.Parameter(torch.Tensor(576,64))
        nn.init.kaiming_uniform_(self.conv_kernel)
    def forward(self, a):
        a_ = F.pad(a, (1,1,1,1), mode='reflect')
        a__ = self.unfold(a_)
        a__ = a__.transpose(1,2)
        a_std = torch.std(a__,dim=-1,keepdim=True)
        a_mean = torch.mean(a__,dim=-1,keepdim=True)
        a____ = (a__ - a_mean)/a_std
        a_out = torch.matmul(a____,self.conv_kernel)
        a_out = self.fold(a_out.transpose(1,2))
        
        return a_out
class norm_conv2(nn.Module):
    def __init__(self,):
        super(norm_conv2, self).__init__()
        self.unfold = torch.nn.Unfold(kernel_size=(3, 3), stride=1)
        self.fold = torch.nn.Fold(output_size=(32, 32), kernel_size=(1,1), stride=1)
        self.conv_kernel = nn.Parameter(torch.Tensor(27,64))
        nn.init.kaiming_uniform_(self.conv_kernel)
    def forward(self, a):
        a_ = F.pad(a, (1,1,1,1), mode='reflect')
        a__ = self.unfold(a_)
        a__ = a__.transpose(1,2)
        a_std = torch.std(a__,dim=-1,keepdim=True)
        a_mean = torch.mean(a__,dim=-1,keepdim=True)
        a____ = (a__ - a_mean)/a_std
        a_out = torch.matmul(a____,self.conv_kernel)
        a_out = self.fold(a_out.transpose(1,2))
        
        return a_out
class ResNetCifar(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 inplanes=64,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetCifar, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.interplanes = [
            self.inplanes, self.inplanes * 2, self.inplanes * 4,
            self.inplanes * 8
        ]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None '
                             'or a 3-element tuple, got {}'.format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv2 = nn.Sequential(
            norm_conv(),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes), nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(block, self.interplanes[0], layers[0])
        self.layer2 = self._make_layer(block,
                                       self.interplanes[1],
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       self.interplanes[2],
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       self.interplanes[3],
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.interplanes[3] * block.expansion, num_classes)
        self.cc = convconv()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        #x = self.conv2(x)
        #x = self.cc(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if isinstance(x,tuple):
            x = x[0]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnetcifar(arch, block, layers, pretrained, **kwargs):
    model = ResNetCifar(block, layers, **kwargs)
    # only load state_dict()
    if pretrained:
        load_state_dict(
            torch.load(model_urls[arch], map_location=torch.device('cpu')),
            model)

    return model


def resnet18cifar(pretrained=False, **kwargs):
    return _resnetcifar('resnet18cifar', BasicBlock, [2, 2, 2, 2], pretrained,
                        **kwargs)


def resnet34cifar(pretrained=False, **kwargs):
    return _resnetcifar('resnet34cifar', BasicBlock, [3, 4, 6, 3], pretrained,
                        **kwargs)


def resnet50cifar(pretrained=False, **kwargs):
    return _resnetcifar('resnet50cifar', Bottleneck, [3, 4, 6, 3], pretrained,
                        **kwargs)


def resnet101cifar(pretrained=False, **kwargs):
    return _resnetcifar('resnet101cifar', Bottleneck, [3, 4, 23, 3],
                        pretrained, **kwargs)


def resnet152cifar(pretrained=False, **kwargs):
    return _resnetcifar('resnet152cifar', Bottleneck, [3, 8, 36, 3],
                        pretrained, **kwargs)


if __name__ == '__main__':
    net = ResNetCifar(Bottleneck, [3, 4, 6, 3], num_classes=1000)
    image_h, image_w = 32, 32
    from thop import profile
    from thop import clever_format
    flops, params = profile(net,
                            inputs=(torch.randn(1, 3, image_h, image_w), ),
                            verbose=False)
    flops, params = clever_format([flops, params], '%.3f')
    out = net(torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    print(f'1111, flops: {flops}, params: {params},out_shape: {out.shape}')
