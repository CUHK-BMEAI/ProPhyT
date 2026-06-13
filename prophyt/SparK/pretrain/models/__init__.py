# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from timm import create_model
from timm.loss import SoftTargetCrossEntropy
from timm.models.layers import drop


from models.convnext import ConvNeXt
from models.resnet import ResNet
from models.resnet_4ch import resnet50_4ch, resnet101_4ch  # 4-channel ResNet for CTP
from models.resnet_1ch import resnet50_1ch, resnet101_1ch  # 1-channel ResNet for NCCT
from models.custom import YourConvNet
_import_resnets_for_timm_registration = (ResNet,)


# log more
def _ex_repr(self):
    return ', '.join(
        f'{k}=' + (f'{v:g}' if isinstance(v, float) else str(v))
        for k, v in vars(self).items()
        if not k.startswith('_') and k != 'training'
        and not isinstance(v, (torch.nn.Module, torch.Tensor))
    )
for clz in (torch.nn.CrossEntropyLoss, SoftTargetCrossEntropy, drop.DropPath):
    if hasattr(clz, 'extra_repr'):
        clz.extra_repr = _ex_repr
    else:
        clz.__repr__ = lambda self: f'{type(self).__name__}({_ex_repr(self)})'


pretrain_default_model_kwargs = {
    'your_convnet': dict(),
    'resnet50': dict(drop_path_rate=0.05),
    'resnet101': dict(drop_path_rate=0.08),
    'resnet152': dict(drop_path_rate=0.10),
    'resnet200': dict(drop_path_rate=0.15),
    'resnet50_4ch': dict(drop_path_rate=0.05),  # 4-channel ResNet50 for CTP
    'resnet101_4ch': dict(drop_path_rate=0.08),  # 4-channel ResNet101 for CTP
    'resnet50_1ch': dict(drop_path_rate=0.05),  # 1-channel ResNet50 for NCCT
    'resnet101_1ch': dict(drop_path_rate=0.08),  # 1-channel ResNet101 for NCCT
    'convnext_small': dict(sparse=True, drop_path_rate=0.2),
    'convnext_base': dict(sparse=True, drop_path_rate=0.3),
    'convnext_large': dict(sparse=True, drop_path_rate=0.4),
}
for kw in pretrain_default_model_kwargs.values():
    kw['pretrained'] = False
    kw['num_classes'] = 0
    kw['global_pool'] = ''


def build_sparse_encoder(name: str, input_size: int, sbn=False, drop_path_rate=0.0, verbose=False):
    from encoder import SparseEncoder
    
    kwargs = pretrain_default_model_kwargs[name]
    if drop_path_rate != 0:
        kwargs['drop_path_rate'] = drop_path_rate
    print(f'[build_sparse_encoder] model kwargs={kwargs}')
    cnn = create_model(name, **kwargs)
    
    return SparseEncoder(cnn, input_size=input_size, sbn=sbn, verbose=verbose)

