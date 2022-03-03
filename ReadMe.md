# Official implementation of paper Importance-based down sampling augments image classification and object detection tasks in convolutional neural networks
## Credits
Credits to zgcr. (2020). pytorch-ImageNet-CIFAR-COCO-VOC-training. Github.[https://github.com/zgcr/simpleAICV-pytorch-ImageNet-COCO-training] for his outstanding work!

## Usage

Please refer to https://github.com/zgcr/simpleAICV-pytorch-ImageNet-COCO-training for details.

For classification, You can reproduce our results on IDS-ResNet, IDS-ResNext, IDS-ResNext by,

cd classification_training/imagenet/resnet_wresnet_resnext_ids_train_example
```
./train.sh
Changing $ in network = '$', Line 25 to different name can configure experiments on other networks.
```
For detection on VOC datasets, you can try,
```
cd detection_training/voc/centernet_res18_resize400_multi_ciou
./train.sh
```
For detection on COCO datasets,
```
cd detection_training/coco/centernet_res18_ids_yoloresize512_multi
./train.sh
```