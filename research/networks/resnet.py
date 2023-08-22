import math

import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models as vision_models
from torchvision import transforms


class RobomimicEncoder(torch.nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        pretrain="supervised",
        num_kp=64,
        freeze_resnet=False,
        backbone=18,
    ):
        super().__init__()
        assert len(observation_space.shape) == 3

        self.normlayer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
        )

        input_channel = observation_space.shape[0]
        self.resnet = ResNet(input_channel=input_channel, backbone=backbone, pretrain=pretrain)
        resnet_output_shape = self.resnet.output_shape(observation_space.shape)
        self.spatial_softmax = SpatialSoftmax(input_shape=resnet_output_shape, num_kp=num_kp)
        spatial_softmax_output_shape = self.spatial_softmax.output_shape(resnet_output_shape)
        self.repr_dim = np.prod(spatial_softmax_output_shape)
        self.flatten = nn.Flatten()

        if freeze_resnet:
            self.resnet.requires_grad_(False)
            if input_channel != 3:
                conv1 = self.resnet.nets[0]
                conv1.requires_grad_(True)  # since this is not pretrained

    def forward(self, obs):
        obs = obs.float() / 255.0
        obs = self.normlayer(obs)
        h = self.resnet(obs)
        h = self.spatial_softmax(h)
        h = self.flatten(h)
        return h

    @property
    def output_space(self):
        return gym.spaces.Box(shape=(self.repr_dim,), low=-np.inf, high=np.inf, dtype=np.float32)


class ResNet(torch.nn.Module):
    """
    A ResNet block that can be used to process input images.
    """

    def __init__(
        self,
        input_channel: int = 3,
        backbone: int = 18,
        pretrain: str = "supervised",
    ):
        """
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            pretrained (bool): if True, load pretrained weights for all ResNet layers.
            input_coord_conv (bool): if True, use a coordinate convolution for the first layer
                (a convolution where input channels are modified to encode spatial pixel location)
        """
        super().__init__()
        model_cls, weights = {
            18: (vision_models.resnet18, vision_models.ResNet18_Weights.DEFAULT),
            34: (vision_models.resnet34, vision_models.ResNet34_Weights.DEFAULT),
            50: (vision_models.resnet50, vision_models.ResNet50_Weights.DEFAULT),
        }[backbone]
        net = model_cls(weights=weights)

        if input_channel != 3:
            net.conv1 = nn.Conv2d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # cut the last fc layer
        self._input_channel = input_channel
        self.nets = torch.nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, inputs):
        x = self.nets(inputs)
        if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
            raise ValueError(
                "Size mismatch: expect size %s, but got size %s"
                % (str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
            )
        return x

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert len(input_shape) == 3
        out_h = int(math.ceil(input_shape[1] / 32.0))
        out_w = int(math.ceil(input_shape[2] / 32.0))
        return [512, out_h, out_w]


class SpatialSoftmax(torch.nn.Module):
    """
    Spatial Softmax Layer.
    Based on Deep Spatial Autoencoders for Visuomotor Learning by Finn et al.
    https://rll.berkeley.edu/dsae/dsae.pdf
    """

    def __init__(
        self,
        input_shape,
        num_kp=None,
        temperature=1.0,
        learnable_temperature=False,
        output_variance=False,
        noise_std=0.0,
    ):
        """
        Args:
            input_shape (list): shape of the input feature (C, H, W)
            num_kp (int): number of keypoints (None for not use spatialsoftmax)
            temperature (float): temperature term for the softmax.
            learnable_temperature (bool): whether to learn the temperature
            output_variance (bool): treat attention as a distribution, and compute second-order statistics to return
            noise_std (float): add random spatial noise to the predicted keypoints
        """
        super(SpatialSoftmax, self).__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape  # (C, H, W)

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._num_kp = num_kp
        else:
            self.nets = None
            self._num_kp = self._in_c
        self.learnable_temperature = learnable_temperature
        self.output_variance = output_variance
        self.noise_std = noise_std

        if self.learnable_temperature:
            # temperature will be learned
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
            self.register_parameter("temperature", temperature)
        else:
            # temperature held constant after initialization
            temperature = torch.nn.Parameter(torch.ones(1) * temperature, requires_grad=False)
            self.register_buffer("temperature", temperature)

        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(1, self._in_h * self._in_w)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, self._in_h * self._in_w)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert len(input_shape) == 3
        assert input_shape[0] == self._in_c
        return [self._num_kp, 2]

    def forward(self, feature):
        """
        Forward pass through spatial softmax layer. For each keypoint, a 2D spatial
        probability distribution is created using a softmax, where the support is the
        pixel locations. This distribution is used to compute the expected value of
        the pixel location, which becomes a keypoint of dimension 2. K such keypoints
        are created.
        Returns:
            out (torch.Tensor or tuple): mean keypoints of shape [B, K, 2], and possibly
                keypoint variance of shape [B, K, 2, 2] corresponding to the covariance
                under the 2D spatial softmax distribution
        """
        assert feature.shape[1] == self._in_c
        assert feature.shape[2] == self._in_h
        assert feature.shape[3] == self._in_w
        if self.nets is not None:
            feature = self.nets(feature)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        feature = feature.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(feature / self.temperature, dim=-1)
        # [1, H * W] x [B * K, H * W] -> [B * K, 1] for spatial coordinate mean in x and y dimensions
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        # stack to [B * K, 2]
        expected_xy = torch.cat([expected_x, expected_y], 1)
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._num_kp, 2)

        if self.training:
            noise = torch.randn_like(feature_keypoints) * self.noise_std
            feature_keypoints += noise

        if self.output_variance:
            # treat attention as a distribution, and compute second-order statistics to return
            expected_xx = torch.sum(self.pos_x * self.pos_x * attention, dim=1, keepdim=True)
            expected_yy = torch.sum(self.pos_y * self.pos_y * attention, dim=1, keepdim=True)
            expected_xy = torch.sum(self.pos_x * self.pos_y * attention, dim=1, keepdim=True)
            var_x = expected_xx - expected_x * expected_x
            var_y = expected_yy - expected_y * expected_y
            var_xy = expected_xy - expected_x * expected_y
            # stack to [B * K, 4] and then reshape to [B, K, 2, 2] where last 2 dims are covariance matrix
            feature_covar = torch.cat([var_x, var_xy, var_xy, var_y], 1).reshape(-1, self._num_kp, 2, 2)
            feature_keypoints = (feature_keypoints, feature_covar)

        return feature_keypoints
