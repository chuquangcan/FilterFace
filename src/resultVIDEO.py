import math
from collections import OrderedDict

import cv2
import numpy as np
import torch.nn.modules
from PIL import Image
from scipy.spatial import Delaunay
from torchvision import transforms

from src.Check_filter_img import get_filter
from src.models.components.res_net_18 import ResNet

if __name__ == "__main__":
    # Chuẩn hóa và transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    sigma = 50
    is_first_frame = True
    simple_transform = transforms.Compose([
        transforms.Resize(size=(256, 256)),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Lấy filter
    filters = get_filter(
        "C:\\Users\\ADMIN\\PycharmProjects\\lightning-hydra-template\\data\\joker.csv")
    filter_img = cv2.imread("C:\\Users\\ADMIN\\PycharmProjects\\lightning-hydra-template\\data\\joker.png")

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

    # Open the video file
    video_path = 'C:\\Users\\ADMIN\\PycharmProjects\\lightning-hydra-template\\src\\InputImageFinal\\checkvideo3.mp4'  # replace with your video file path
    cap = cv2.VideoCapture(0)

    # Get the video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a VideoWriter object to save the output video
    output_path = 'C:\\Users\\ADMIN\\PycharmProjects\\lightning-hydra-template\\src\\InputImageFinal\\video3.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    change_filters = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no frames are read
        image = Image.fromarray(frame)
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
        y = modelLM(simple_transform(cropped_image).unsqueeze(dim=0)).detach()
        y = y[0]
        y = (y + 0.5) * np.array([w * 224 / 256, h * 224 / 256]) + np.array(
            [w * 16 / 256, h * 16 / 224])
        y /= torch.tensor([w / w1, h / h1])
        y += bounding_box[0]
        y = y.numpy().astype(np.int32)

        # optical flow va on dinh keypoints
        img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if is_first_frame:
            img2GrayPrev = np.copy(img2Gray)
            yPrev = np.array(y, np.float32)
            is_first_frame = False

        lk_params = dict(winSize=(101, 101), maxLevel=15,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
        yNext, st, err = cv2.calcOpticalFlowPyrLK(img2GrayPrev, img2Gray, yPrev,
                                                  np.array(y, np.float32),
                                                  **lk_params)
        for i in range(y.shape[0]):
            d = cv2.norm(np.array(y[i]) - yNext[i])
            alpha = math.exp(-d * d / sigma)
            y[i] = (1 - alpha) * np.array(y[i]) + alpha * yNext[i]
            y[i] = (int(y[i, 0]), int(y[i, 1]))

        yPrev = np.array(y, np.float32)
        img2GrayPrev = img2Gray

        triangulation = Delaunay(y)

        if (change_filters):
            # dan filter sang mat
            for i in triangulation.simplices:
                tripoint1 = y[i].reshape((-1, 1, 2))
                r1 = cv2.boundingRect(tripoint1)
                tripoint2 = filters[i].reshape((-1, 1, 2))
                r2 = cv2.boundingRect(tripoint2)
                cropped_filter_img = filter_img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
                mask = np.zeros((r1[3], r1[2], 3), dtype=np.uint8)
                tripoints2 = []
                tripoints1 = []
                for k in range(3):
                    tripoints2.append(((tripoint2[k][0][0] - r2[0]), (tripoint2[k][0][1] - r2[1])))
                    tripoints1.append(((tripoint1[k][0][0] - r1[0]), (tripoint1[k][0][1] - r1[1])))

                # scale cho filter giong mat
                warpMat = cv2.getAffineTransform(np.float32(tripoints2), np.float32(tripoints1))
                cropped_filter_img = cv2.warpAffine(cropped_filter_img, warpMat, (r1[2], r1[3]), None,
                                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                cv2.fillConvexPoly(mask, np.int32(tripoints1), (255, 255, 255), 8, 0)
                cropped_filter_img = cv2.bitwise_and(cropped_filter_img, mask)

                # dan filter vao mat
                roi = frame[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
                ret, mask2 = cv2.threshold(cv2.cvtColor(cropped_filter_img, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask2)
                frame[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]] = cv2.bitwise_and(roi, roi, mask=mask_inv)
                frame[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]] += cv2.bitwise_and(
                    cropped_filter_img, cropped_filter_img, mask=mask2)
        else:
            # Draw the points on the frame
            for i in range(y.shape[0]):
                if (i == 30):
                    cv2.circle(frame, y[i], 2, (0, 0, 225), -1)  # assuming red color
                else:
                    cv2.circle(frame, y[i], 2, (0, 255, 0), -1)  # assuming green color

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (int(bounding_box[0][0]), int(bounding_box[0][1])),
                          (int(bounding_box[2][0]), int(bounding_box[2][1])), (0, 255, 0),
                          2)  # Màu xanh lá cây, độ dày đường vẽ là 2 pixel

            # Draw Delaunay triangle on the frame
            list_triangle = []
            for i in triangulation.simplices:
                triangle_vertices_int = y[i].reshape((-1, 1, 2))
                list_triangle.append(triangle_vertices_int)
            cv2.polylines(frame, list_triangle, isClosed=True, color=(0, 255, 0), thickness=1)

        cv2.imshow('Video Feed', frame)
        out.write(frame)  # Write frame to the output video

        # Kiểm tra xem có phím nào được nhấn hay không
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('f'):
            change_filters = not change_filters
    # Release video capture and writer objects
    cap.release()
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
