import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_resnet50

# from torchvision.models.segmentation import deeplabv3_resnet101


def prepare_model(num_classes=2):
    model = deeplabv3_resnet50(weights='DEFAULT')
    # just replacing some final layers
    model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    return model


# resnet 101 here if we want to try

# def prepare_model(num_classes=2):
#     model = deeplabv3_resnet101(weights='DEFAULT')
#     model.classifier[4] = nn.Conv2d(256, num_classes, 1)
#     model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
#     return model