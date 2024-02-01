import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import PIL.Image
from scipy.spatial import Delaunay


def get_filter(file_csv_path: str):
    df = pd.read_csv(file_csv_path)
    PX = df['PX'].values
    PY = df['PY'].values
    keypoints = []
    for i in range(PX.shape[0]):
        keypoints.append([PX[i], PY[i]])
    keypoints = np.array(keypoints)
    return keypoints


if __name__ == "__main__":
    file_path = 'C:\\Users\\ADMIN\\PycharmProjects\\lightning-hydra-template\\data\\labels_my-project-name_2024-01-25-03-50-10.csv'
    df = pd.read_csv(file_path)
    PX = df['PX'].values
    PY = df['PY'].values
    keypoints = []
    for i in range(PX.shape[0]):
        keypoints.append([PX[i], PY[i]])
    keypoints = np.array(keypoints)
    triangle = Delaunay(keypoints)
    image = cv2.imread(
        "C:\\Users\\ADMIN\\PycharmProjects\\lightning-hydra-template\\data\\klipartz.com_digital_art_x4.png")
    triangle1 = triangle.simplices[0]
    triangle2 = triangle.simplices[10]
    r1 = [min(keypoints[triangle1[0]][0], keypoints[triangle1[1]][0], keypoints[triangle1[2]][0]),
          min(keypoints[triangle1[0]][1], keypoints[triangle1[1]][1], keypoints[triangle1[2]][1])]
    r2 = [max(keypoints[triangle1[0]][0], keypoints[triangle1[1]][0], keypoints[triangle1[2]][0]),
          max(keypoints[triangle1[0]][1], keypoints[triangle1[1]][1], keypoints[triangle1[2]][1])]
    r3 = [min(keypoints[triangle2[0]][0], keypoints[triangle2[1]][0], keypoints[triangle2[2]][0]),
          min(keypoints[triangle2[0]][1], keypoints[triangle2[1]][1], keypoints[triangle2[2]][1])]
    r4 = [max(keypoints[triangle2[0]][0], keypoints[triangle2[1]][0], keypoints[triangle2[2]][0]),
          max(keypoints[triangle2[0]][1], keypoints[triangle2[1]][1], keypoints[triangle2[2]][1])]
    print(r3)
    print(r4)
    cropped_img1 = image[r1[1]:r2[1], r1[0]:r2[0]]
    cropped_img2 = image[r3[1]:r4[1], r3[0]:r4[0]]
    res = image
    mask1 = np.zeros((r2[1] - r1[1], r2[0] - r1[0], 3), dtype=np.uint8)
    mask2 = np.zeros((r4[1] - r3[1], r4[0] - r3[0], 3), dtype=np.uint8)
    triangle_points1 = []
    triangle_points2 = []
    for i in range(3):
        triangle_points1.append(keypoints[triangle1[i]] - r1)
    for i in range(3):
        triangle_points2.append(keypoints[triangle2[i]] - r3)
    # warpMat = cv2.getAffineTransform(np.float32(triangle_points1), np.float32(triangle_points2))
    # cropped_img = cv2.warpAffine(cropped_img, warpMat, (r3[1]-r4[1], r3[0]-r4[0]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    # cv2.fillConvexPoly(mask, np.int32(triangle_points2), (255, 255, 255), 8, 0)
    # cropped_img = cv2.bitwise_and(cropped_img, mask)
    warpMat = cv2.getAffineTransform(np.float32(triangle_points1), np.float32(triangle_points2))
    cropped_img1 = cv2.warpAffine(cropped_img1, warpMat, (r4[0]-r3[0], r4[1]-r3[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    cv2.fillConvexPoly(mask2, np.int32(triangle_points2), (255, 255, 255), 8, 0)
    cropped_img1 = cv2.bitwise_and(cropped_img1, mask2)
    roi = res[r3[1]:r4[1], r3[0]:r4[0]]
    ret, mask = cv2.threshold(cv2.cvtColor(cropped_img1, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    res[r3[1]:r4[1], r3[0]:r4[0]] = cv2.bitwise_and(roi, roi, mask=mask_inv)
    res[r3[1]:r4[1], r3[0]:r4[0]] += cv2.bitwise_and(
        cropped_img1, cropped_img1, mask=mask)

    cv2.imshow("hello", cropped_img1)
    # Lấy kích thước ảnh
    image_height, image_width = res.shape[:2]

    # Tính tỷ lệ để điều chỉnh kích thước ảnh
    scale_factor = min(1000 / image_width, 1000 / image_height)

    # Tính kích thước mới của ảnh
    new_width = int(image_width * scale_factor)
    new_height = int(image_height * scale_factor)

    # Điều chỉnh kích thước ảnh
    resized_image = cv2.resize(image, (new_width, new_height))
    cv2.imshow("hello2", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # plt.imshow(image)
    # plt.triplot(keypoints[:,0], keypoints[:, 1], triangle.simplices, c='green', marker='o')
    # plt.show()
