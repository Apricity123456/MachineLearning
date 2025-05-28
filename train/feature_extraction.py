import os
import cv2
import numpy as np
import pandas as pd

# 获取当前脚本所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接图像和 mask 文件夹的绝对路径
image_dir = os.path.join(base_dir, 'data', 'images')
mask_dir = os.path.join(base_dir, 'data', 'masks')

# 获取所有图像文件
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.JPG') or f.endswith('.jpg')])

def extract_features(image_path, mask_path):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 二值化 mask
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 获取昆虫区域
    total_pixels = img.shape[0] * img.shape[1]
    bug_pixels = np.sum(mask_bin > 0)
    if bug_pixels == 0:
        return None  # 忽略空 mask

    area_ratio = bug_pixels / total_pixels

    # 获取 RGB
    bug_rgb = img[mask_bin > 0]
    R, G, B = bug_rgb[:, 2], bug_rgb[:, 1], bug_rgb[:, 0]

    # RGB统计特征
    R_mean, G_mean, B_mean = np.mean(R), np.mean(G), np.mean(B)
    R_std, G_std, B_std = np.std(R), np.std(G), np.std(B)
    R_min, G_min, B_min = np.min(R), np.min(G), np.min(B)
    R_max, G_max, B_max = np.max(R), np.max(G), np.max(B)

    # 找轮廓（只取最大轮廓）
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
    roughness = (perimeter ** 2) / (4 * np.pi * area + 1e-6)

    # 对称性（左右翻转）
    h_center = mask_bin.shape[1] // 2
    left = mask_bin[:, :h_center]
    right = mask_bin[:, -h_center:]
    flipped_right = cv2.flip(right, 1)
    min_shape = min(left.shape[1], flipped_right.shape[1])
    symmetry_diff = np.sum(np.abs(left[:, :min_shape] - flipped_right[:, :min_shape]))
    symmetry_score = 1 - (symmetry_diff / bug_pixels)

    # Hu Moments
    moments = cv2.moments(mask_bin)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments_log = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    return {
        'area_ratio': area_ratio,
        'R_mean': R_mean, 'G_mean': G_mean, 'B_mean': B_mean,
        'R_std': R_std, 'G_std': G_std, 'B_std': B_std,
        'R_min': R_min, 'G_min': G_min, 'B_min': B_min,
        'R_max': R_max, 'G_max': G_max, 'B_max': B_max,
        'aspect_ratio': aspect_ratio,
        'circularity': circularity,
        'roughness': roughness,
        'symmetry_score': symmetry_score,
        **{f'hu_{i+1}': hu_moments_log[i] for i in range(7)}
    }

# 主执行
all_features = []

for fname in image_files:
    img_id = os.path.splitext(fname)[0]
    image_path = os.path.join(image_dir, fname)
    mask_path = os.path.join(mask_dir, f"binary_{img_id}.tif")

    if not os.path.exists(mask_path):
        print(f"[跳过] 未找到 mask: {mask_path}")
        continue

    features = extract_features(image_path, mask_path)
    if features:
        features['ID'] = int(os.path.splitext(fname)[0])
        all_features.append(features)

# 保存为 CSV
df = pd.DataFrame(all_features)
df.to_csv('train/features_with_shape.csv', index=False)
print("✅ 特征提取完成，保存为 features_with_shape.csv")
