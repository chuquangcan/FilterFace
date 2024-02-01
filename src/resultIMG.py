import os
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import Delaunay
from tensorflow import Tensor
from torch import nn
import torch.nn.modules
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

from src.models.components.res_net_18 import ResNet


def draw_landmarks(image: Image, input: torch.Tensor, output: torch.Tensor, box_top_left: torch.Tensor,
                   cropped_image: Image) -> Image:
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
    landmarks /= torch.tensor([width / width_c, height / height_c])
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
checkpointLM_path = 'C:\\Users\\ADMIN\\PycharmProjects\\lightning-hydra-template\\logs\\train\\runs\\bestADAMLANDMARKS4\\checkpoints\\epoch_075.ckpt'
modelLM = ResNet(noLayers=18)
checkpointBB_path = 'C:\\Users\\ADMIN\\PycharmProjects\\lightning-hydra-template\\logs\\train\\runs\\bestADAMBOUNDINGBOX2\\checkpoints\\epoch_057.ckpt'
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

# annotated_input_image = draw_landmarks(image, image_tensor, y, bounding_box[0], cropped_image).resize((w, h))
# annotated_input_image.save(
#     "C:\\Users\\ADMIN\\PycharmProjects\\lightning-hydra-template\\src\\InputImageFinal\\res2.png")

y = y.detach()
y = y[0]
y = (y + 0.5) * np.array([w * 224 / 256, h * 224 / 256]) + np.array(
    [w * 16 / 256, h * 16 / 224])
y /= torch.tensor([w / w1, h / h1])
y += bounding_box[0]
triangulation = Delaunay(y)
print(triangulation.simplices[0])
# Plot the original points
plt.scatter(y[:, 0], y[:, 1], c='red', marker='o', label='Points')
plt.imshow(image)
# Plot the Delaunay triangles
plt.triplot(y[:, 0], y[:, 1], triangulation.simplices, c='blue', label='Delaunay Triangulation')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Delaunay Triangulation')
plt.legend()
plt.show()
