import os

import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms
from typing import Type, Any, Callable, Union, List, Optional, Dict

import torchvision.transforms.functional
import comfy.model_management

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            use_last_fc: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.use_last_fc = use_last_fc
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.use_last_fc:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        if self.use_last_fc:
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class Estimate3DV(nn.Module):
    fc_dim = 257
    camera_distance = 10.
    center = 112.

    def __init__(self, checkpoint, use_last_fc=False):
        super(Estimate3DV, self).__init__()
        self.use_last_fc = use_last_fc

        last_dim = 2048
        backbone = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], use_last_fc=use_last_fc, num_classes=self.fc_dim)
        self.backbone = backbone
        if not use_last_fc:
            self.final_layers = nn.ModuleList([
                conv1x1(last_dim, 80, bias=True),  # id layer
                conv1x1(last_dim, 64, bias=True),  # exp layer
                conv1x1(last_dim, 80, bias=True),  # tex layer
                conv1x1(last_dim, 3, bias=True),  # angle layer
                conv1x1(last_dim, 27, bias=True),  # gamma layer
                conv1x1(last_dim, 2, bias=True),  # tx, ty
                conv1x1(last_dim, 1, bias=True)  # tz
            ])

        self.register_buffer('mean_shape', torch.zeros(size=(107127, 1), dtype=torch.float32))
        self.register_buffer('id_base', torch.zeros(size=(107127, 80), dtype=torch.float32))
        self.register_buffer('exp_base', torch.zeros(size=(107127, 64), dtype=torch.float32))
        self.register_buffer('kps_ind', torch.zeros(size=(68,), dtype=torch.int64))
        self.register_buffer('persc_proj', torch.zeros(size=(3, 3), dtype=torch.float32))

        state_dict = torch.load(checkpoint, map_location='cpu')
        self.load_state_dict(state_dict)
        self.eval()
        self.device = comfy.model_management.get_torch_device()
        self.to(comfy.model_management.get_torch_device())

    @torch.inference_mode()
    def estimate(self, x: Tensor) -> List:
        x = x.to(comfy.model_management.get_torch_device())
        batch_size = x.shape[0]
        x = self.backbone(x)
        if not self.use_last_fc:
            output = []
            for layer in self.final_layers:
                output.append(layer(x))
            x = torch.flatten(torch.cat(output, dim=1), 1)
        coeffs = []
        for i in range(batch_size):
            coeffs.append({
                'id': x[i:i+1, :80],
                'exp': x[i:i+1, 80:144],
                'tex': x[i:i+1, 144:224],
                'angle': x[i:i+1, 224:227],
                'gamma': x[i:i+1, 227:254],
                'trans': x[i:i+1, 254:]
            })
        return coeffs

    def compute_shape(self, id_coeff, exp_coeff):
        batch_size = id_coeff.shape[0]
        id_part = torch.einsum('ij,aj->ai', self.id_base, id_coeff)
        exp_part = torch.einsum('ij,aj->ai', self.exp_base, exp_coeff)
        face_shape = id_part + exp_part + self.mean_shape.reshape([1, -1])

        return face_shape.reshape([batch_size, -1, 3])

    def compute_rotation(self, angles):
        batch_size = angles.shape[0]
        ones = torch.ones([batch_size, 1]).to(self.device)
        zeros = torch.zeros([batch_size, 1]).to(self.device)
        x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

        value_list = [
            ones, zeros, zeros, zeros,
            torch.cos(x), -torch.sin(x), zeros,
            torch.sin(x),
            torch.cos(x)
        ]
        rot_x = torch.cat(value_list, dim=1).reshape([batch_size, 3, 3])

        value_list = [
            torch.cos(y), zeros,
            torch.sin(y), zeros, ones, zeros, -torch.sin(y), zeros,
            torch.cos(y)
        ]
        rot_y = torch.cat(value_list, dim=1).reshape([batch_size, 3, 3])

        value_list = [
            torch.cos(z), -torch.sin(z), zeros,
            torch.sin(z),
            torch.cos(z), zeros, zeros, zeros, ones
        ]
        rot_z = torch.cat(value_list, dim=1).reshape([batch_size, 3, 3])

        rot = rot_z @ rot_y @ rot_x
        return rot.permute(0, 2, 1)

    def to_camera(self, face_shape):
        face_shape[..., -1] = self.camera_distance - face_shape[..., -1]
        return face_shape

    def to_image(self, face_shape):
        face_proj = face_shape @ self.persc_proj
        face_proj = face_proj[..., :2] / face_proj[..., 2:]

        return face_proj

    def transform(self, face_shape, rot, trans):
        return face_shape @ rot + trans.unsqueeze(1)

    def get_landmarks(self, face_proj):
        return face_proj[:, self.kps_ind]

    def coeffs_parse(self, coeffs):
        '''
        :param coeffs: bachsize only for 1
        :return: landmarks (1*68*3), landmarks (1*224*224)
        '''
        face_src_shape = self.compute_shape(coeffs['id'], coeffs['exp'])

        rotation = self.compute_rotation(coeffs['angle'])
        face_shape_transformed = self.transform(face_src_shape, rotation,
                                                     coeffs['trans'])
        face_vertex = self.to_camera(face_shape_transformed)

        face_proj = self.to_image(face_vertex)

        face_proj[..., -1] = self.center * 2 - face_proj[..., -1]

        landmark = self.get_landmarks(face_proj)

        mouth_lmk = landmark[0, 49:].int()

        face_proj_coord = face_proj.squeeze(0).int()

        mask_zeros = torch.zeros(size=(1, int(self.center * 2), int(self.center * 2)),
                                 device=face_proj.device, dtype=torch.float32)

        mask_zeros[0, mouth_lmk[:, 1], mouth_lmk[:, 0]] = 1

        mask_zeros = torchvision.transforms.functional.gaussian_blur(mask_zeros, [15, 15], [1.0, 1.0])

        mask_zeros[mask_zeros != 0] = 1

        mask_zeros[0, face_proj_coord[:, 1], face_proj_coord[:, 0]] = 1

        mask = torchvision.transforms.functional.gaussian_blur(mask_zeros, [3, 3], [1.0, 1.0])
        mask[mask != 0] = 1
        mask = mask*255
        mask = mask.to(torch.uint8)
        return landmark, mask
