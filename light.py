import os
import cv2
import numpy as np

def compute_gray_mean(image):
    """基于灰度加权平均的亮度"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def compute_hsv_v_mean(image):
    """基于 HSV 中 V 通道的亮度"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]  # V通道
    return np.mean(v)

def analyze_directory(image_dir):
    gray_means = []
    v_means = []
    image_names = []

    for file_name in os.listdir(image_dir):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
            image_path = os.path.join(image_dir, file_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            gray_mean = compute_gray_mean(image)
            v_mean = compute_hsv_v_mean(image)

            gray_means.append(gray_mean)
            v_means.append(v_mean)
            image_names.append(file_name)

    gray_means = np.array(gray_means)
    print(np.mean(gray_means))
    v_means = np.array(v_means)
    print(v_means)

    # 阈值 = 均值 - 1×标准差
    gray_thresh = np.mean(gray_means) - np.std(gray_means)
    v_thresh = np.mean(v_means) - np.std(v_means)

    # 筛选低亮度图像
    gray_low_light = [name for name, val in zip(image_names, gray_means) if val < gray_thresh]
    v_low_light = [name for name, val in zip(image_names, v_means) if val < v_thresh]

    print("=== 统计结果 ===")
    print(f"共处理图像数量: {len(image_names)}")
    print(f"灰度亮度阈值: {gray_thresh:.2f}")
    print(f"V通道亮度阈值: {v_thresh:.2f}")
    print(f"灰度法低亮度图像数: {len(gray_low_light)}")
    print(f"HSV法低亮度图像数: {len(v_low_light)}")

    # 保存结果
    with open("low_light_images_gray.txt", "w") as f:
        f.writelines([name + "\n" for name in gray_low_light])
    with open("low_light_images_hsv.txt", "w") as f:
        f.writelines([name + "\n" for name in v_low_light])

    return gray_means, v_means, image_names

# 设置图像路径
image_directory = r"E:\airport_data\test\ychdata\images"
gray_means, v_means, names = analyze_directory(image_directory)
