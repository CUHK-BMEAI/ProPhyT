# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.resnet import ResNet
from timm.models.registry import register_model


# hack: inject the `get_downsample_ratio` function into `timm.models.resnet.ResNet`
def get_downsample_ratio(self: ResNet) -> int:
    return 32


# hack: inject the `get_feature_map_channels` function into `timm.models.resnet.ResNet`
def get_feature_map_channels(self: ResNet) -> List[int]:
    # `self.feature_info` is maintained by `timm`
    return [info['num_chs'] for info in self.feature_info[1:]]


# hack: override the forward function of `timm.models.resnet.ResNet`
def forward(self, x, hierarchical=False):
    """ this forward function is a modified version of `timm.models.resnet.ResNet.forward`
    >>> ResNet.forward
    """
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.act1(x)
    x = self.maxpool(x)
    
    if hierarchical:
        ls = []
        x = self.layer1(x); ls.append(x)
        x = self.layer2(x); ls.append(x)
        x = self.layer3(x); ls.append(x)
        x = self.layer4(x); ls.append(x)
        return ls
    else:
        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        x = self.fc(x)
        return x


def create_resnet_4ch(base_model_name='resnet50', in_chans=4, **kwargs):
    """
    创建支持4通道输入的ResNet模型
    
    Args:
        base_model_name: 基础ResNet模型名称 (resnet50, resnet101, etc.)
        in_chans: 输入通道数，默认4（CTP参数图）
        **kwargs: 其他模型参数（可能已包含num_classes, global_pool等）
    """
    from timm import create_model
    
    # 确保kwargs中有必要的参数，但不覆盖已存在的
    model_kwargs = {
        'pretrained': False,
        'num_classes': 0,
        'global_pool': '',
    }
    model_kwargs.update(kwargs)  # kwargs中的值会覆盖默认值
    
    # 创建基础模型（3通道）
    model = create_model(base_model_name, **model_kwargs)
    
    # 修改第一层卷积以支持4通道输入
    old_conv1 = model.conv1
    new_conv1 = nn.Conv2d(
        in_chans, 
        old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=old_conv1.bias is not None
    )
    
    # 初始化新卷积层
    # 对于前3个通道，复制原权重；对于第4个通道，使用均值初始化
    with torch.no_grad():
        if old_conv1.weight.shape[1] == 3:
            # 复制前3个通道的权重
            new_conv1.weight[:, :3, :, :].copy_(old_conv1.weight)
            # 第4个通道使用前3个通道的均值
            new_conv1.weight[:, 3:4, :, :].copy_(old_conv1.weight.mean(dim=1, keepdim=True))
        else:
            # 如果输入通道数不匹配，使用kaiming初始化
            nn.init.kaiming_normal_(new_conv1.weight, mode='fan_out', nonlinearity='relu')
        
        if new_conv1.bias is not None:
            if old_conv1.bias is not None:
                new_conv1.bias.copy_(old_conv1.bias)
            else:
                nn.init.constant_(new_conv1.bias, 0)
    
    model.conv1 = new_conv1
    
    # 注入必要的方法
    model.get_downsample_ratio = get_downsample_ratio.__get__(model, type(model))
    model.get_feature_map_channels = get_feature_map_channels.__get__(model, type(model))
    model.forward = forward.__get__(model, type(model))
    
    return model


@register_model
def resnet50_4ch(pretrained=False, **kwargs):
    """ResNet50 with 4-channel input for CTP parameter maps"""
    return create_resnet_4ch('resnet50', in_chans=4, **kwargs)


@register_model
def resnet101_4ch(pretrained=False, **kwargs):
    """ResNet101 with 4-channel input for CTP parameter maps"""
    return create_resnet_4ch('resnet101', in_chans=4, **kwargs)


@torch.no_grad()
def convnet_test():
    from timm.models import create_model
    cnn = create_model('resnet50_4ch')
    print('get_downsample_ratio:', cnn.get_downsample_ratio())
    print('get_feature_map_channels:', cnn.get_feature_map_channels())
    
    downsample_ratio = cnn.get_downsample_ratio()
    feature_map_channels = cnn.get_feature_map_channels()
    
    # check the forward function
    B, C, H, W = 4, 4, 224, 224  # 4 channels for CTP
    inp = torch.rand(B, C, H, W)
    feats = cnn(inp, hierarchical=True)
    assert isinstance(feats, list)
    assert len(feats) == len(feature_map_channels)
    print([tuple(t.shape) for t in feats])
    
    # check the downsample ratio
    feats = cnn(inp, hierarchical=True)
    assert feats[-1].shape[-2] == H // downsample_ratio
    assert feats[-1].shape[-1] == W // downsample_ratio
    
    # check the channel number
    for feat, ch in zip(feats, feature_map_channels):
        assert feat.ndim == 4
        assert feat.shape[1] == ch


if __name__ == '__main__':
    convnet_test()
