import os
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image, ImageDraw
from tensorflow import Tensor
from torch import nn
import torch.nn.modules
from torchvision import transforms
import matplotlib.pyplot as plt


class BasicBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            expansion: int = 1,
            downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        id = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if (self.downsample is not None):
            id = self.downsample(x)

        out += id
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: BasicBlock = BasicBlock,
            channels: int = 3,
            noLayers: int = 34,
            output_shape: list = [68, 2],
    ) -> None:
        super(ResNet, self).__init__()
        self.output_shape = output_shape
        num_classes = output_shape[0] * output_shape[1]
        if noLayers == 18:
            layers = [2, 2, 2, 2]
            self.expansion = 1
        elif noLayers == 34:
            layers = [3, 4, 6, 3]
            self.expansion = 1
        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            in_channels=channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes)

    def _make_layer(
            self,
            block: BasicBlock,
            out_channels: int,
            blocks: int,
            stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    self.expansion
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x.reshape(x.size(0), self.output_shape[0], self.output_shape[1])


def getModelResNet18(channels, num_classes):
    return ResNet(channels, 18, num_classes=num_classes)


if __name__ == "__main__":
    def draw_landmarks(image: Image, input: torch.Tensor, output: torch.Tensor, box_top_left: torch.Tensor, cropped_image: Image) -> Image:
        # remove first dimension (batch_size)
        input = input.squeeze()
        output = output.squeeze()
        input = input.detach()
        output = output.detach()

        # reverse color transform
        def denormalize(input: torch.Tensor, std=std, mean=mean) -> torch.Tensor:
            # clone: make a copy
            tmp = input.clone()

            # denormalize
            for t, m, s in zip(tmp, mean, std):
                t.mul_(s).add_(m)

            # clamp: limit value to [0, 1]
            return torch.clamp(tmp, 0, 1)

        input = denormalize(input)

        # get information of original input
        width, height = image.size
        width_c, height_c = cropped_image.size

        # denormalized output (landmarks)
        landmarks = (output + 0.5) * np.array([width * 224 / 256, height * 224 / 256]) + np.array(
            [width * 16 / 256, height * 16 / 224])
        landmarks /= torch.tensor([width/width_c, height/height_c])
        landmarks += box_top_left
        # draw landmarks on original image
        draw = ImageDraw.Draw(image)
        if (landmarks.shape[0] == 68):
            for x, y in landmarks:
                draw.ellipse([(x - 4, y - 4), (x + 4, y + 4)], fill=(0, 255, 0))
        else:
            draw.rectangle([landmarks[0][0], landmarks[0][1], landmarks[2][0], landmarks[2][1]], outline="green",
                           width=2)
        # return annotated image
        return image


    # Chuẩn hóa và transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    simple_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Đường dẫn đến tệp checkpoint
    checkpointLM_path = 'C:\\Users\\ADMIN\\PycharmProjects\\lightning-hydra-template\\logs\\train\\runs\\bestADAMLANDMARKS1\\checkpoints\\epoch_021.ckpt'
    modelLM = ResNet(noLayers=18)
    checkpointBB_path = 'C:\\Users\\ADMIN\\PycharmProjects\\lightning-hydra-template\\logs\\train\\runs\\bestADAMBOUNDINGBOX1\\checkpoints\\epoch_054.ckpt'
    modelBB = ResNet(noLayers=18, output_shape=[4, 2])

    # Nạp trạng thái từ checkpoint
    checkpointLM = torch.load(checkpointLM_path)
    new_state_dictLM = OrderedDict()
    for k, v in checkpointLM['state_dict'].items():
        name = k.replace('net.', '')  # Thay thế "net." bằng chuỗi rỗng
        new_state_dictLM[name] = v
    modelLM.load_state_dict(new_state_dictLM)

    checkpointBB = torch.load(checkpointBB_path)
    new_state_dictBB = OrderedDict()
    for k, v in checkpointBB['state_dict'].items():
        name = k.replace('net.', '')  # Thay thế "net." bằng chuỗi rỗng
        new_state_dictBB[name] = v
    modelBB.load_state_dict(new_state_dictBB)

    # Load image
    image = Image.open(
        "C:\\Users\\ADMIN\\PycharmProjects\\lightning-hydra-template\\src\\InputImageFinal\\check5.jpg").convert(
        "RGB")
    w, h = image.size
    image_tensor = simple_transform(image)
    image_tensor = image_tensor.unsqueeze(dim=0)

    modelBB.eval()
    bounding_box = modelBB(image_tensor).detach()
    bounding_box = (bounding_box + 0.5) * np.array([w * 224 / 256, h * 224 / 256]) + np.array(
        [w * 16 / 256, h * 16 / 224])
    bounding_box = bounding_box[0]
    if bounding_box[0, 0] < bounding_box[2, 0] and bounding_box[0, 1] < bounding_box[2, 1]:
        cropped_image: Image = image.crop(
            (int(bounding_box[0, 0]), int(bounding_box[0, 1]), int(bounding_box[2, 0]), int(bounding_box[2, 1])))

    w1, h1 = cropped_image.size
    modelLM.eval()
    y = modelLM(simple_transform(cropped_image).unsqueeze(dim=0))
    annotated_input_image = draw_landmarks(image, image_tensor, y, bounding_box[0], cropped_image).resize((w, h))
    annotated_input_image.save(
        "C:\\Users\\ADMIN\\PycharmProjects\\lightning-hydra-template\\src\\InputImageFinal\\res.png")
