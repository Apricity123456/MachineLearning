import os
import cv2
import numpy as np
import pandas as pd
import joblib

# === 设置当前目录 ===
base_dir = os.path.dirname(os.path.abspath(__file__))

# 输入文件名（如：new_001.JPG）和对应mask名（binary_new_001.tif）
image_name = "new_001.JPG"
image_path = os.path.join(base_dir, 'data', 'images', image_name)
mask_path = os.path.join(base_dir, 'data', 'masks', f'binary_{os.path.splitext(image_name)[0]}.tif')

# 加载模型
model_path = os.path.join(base_dir, 'model_rf_multiclass.pkl')

model = joblib.load(model_path)

# 特征提取函数（必须和训练时完全一致）
def extract_features(image_path, mask_path):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    total_pixels = img.shape[0] * img.shape[1]
    bug_pixels = np.sum(mask_bin > 0)
    if bug_pixels == 0:
        return None

    area_ratio = bug_pixels / total_pixels
    bug_rgb = img[mask_bin > 0]
    R, G, B = bug_rgb[:, 2], bug_rgb[:, 1], bug_rgb[:, 0]
    R_mean, G_mean, B_mean = np.mean(R), np.mean(G), np.mean(B)
    R_std, G_std, B_std = np.std(R), np.std(G), np.std(B)
    R_min, G_min, B_min = np.min(R), np.min(G), np.min(B)
    R_max, G_max, B_max = np.max(R), np.max(G), np.max(B)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / h
    circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
    roughness = (perimeter ** 2) / (4 * np.pi * area + 1e-6)

    h_center = mask_bin.shape[1] // 2
    left = mask_bin[:, :h_center]
    right = mask_bin[:, -h_center:]
    flipped_right = cv2.flip(right, 1)
    min_shape = min(left.shape[1], flipped_right.shape[1])
    symmetry_diff = np.sum(np.abs(left[:, :min_shape] - flipped_right[:, :min_shape]))
    symmetry_score = 1 - (symmetry_diff / bug_pixels)

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

# 特征提取并预测
features = extract_features(image_path, mask_path)

if features:
    X_new = pd.DataFrame([features])
    prediction = model.predict(X_new)[0]
    print(f"🐝 预测结果：这是一只 {prediction}")
else:
    print("⚠️ 无法提取特征，请检查图片或mask是否存在有效前景区域。")
