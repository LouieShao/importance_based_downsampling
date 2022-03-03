import os
import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
sys.path.append(BASE_DIR)

from tools.path import ILSVRC2012_path

from simpleAICV.classification import backbones
from simpleAICV.classification import losses

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

class config:
    val_dataset_path = os.path.join(ILSVRC2012_path, 'val_sample')

    network = 'resnet18_sample'
    pretrained = True
    num_classes = 1000
    input_image_size = 224
    scale = 256 / 224

    model = backbones.__dict__[network](**{
        'pretrained': pretrained,
        'num_classes': 1000,
    })
    
    criterion = losses.__dict__['CELoss']()

    val_dataset = datasets.ImageFolder(
        val_dataset_path,
        transforms.Compose([
            transforms.Resize(int(input_image_size * scale)),
            transforms.CenterCrop(input_image_size),
            transforms.ToTensor(),
            #transforms.TenCrop(224),
            #transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))
    
    distributed = True
    seed = 0
    batch_size = 32
    num_workers = 16
    trained_model_path = '/home/shaoshihao/model_zoo/resnet/resnet18-epoch90-acc71.616.pth'
