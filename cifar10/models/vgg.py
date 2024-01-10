from torchvision.models import vgg11
import torch.nn as nn

"""
def VGG11():
    model = vgg11(num_classes=10)
    return model"""





from torch import nn
from torchvision.models import vgg11

class VGG():
    # The resnet model from torchvision are purposed for ImageNet and perform poorly on CIFAR10/100

    def remove_bias(self, model: nn.Module):
        model = model.apply(lambda m: m.register_parameter('bias', None))
        return model

    def get_model(self, *args, **kwargs) -> nn.Module:
        bias = kwargs.pop('bias', False)
        num_channels = kwargs.pop('num_channels', 3)

        model = vgg11(*args, **kwargs)

        if num_channels != 3:
            model.features[0] = nn.Conv2d(num_channels, 64, kernel_size=3, padding=1)

        # Remove bias if it's not an option during model creation
        if not bias:
            model = self.remove_bias(model)

        return model


def VGG11():
    return VGG().get_model(num_classes=10, bias=False)

