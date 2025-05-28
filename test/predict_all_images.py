import os
import argparse
import pandas as pd
import numpy as np
import cv2
import joblib
from sklearn.preprocessing import StandardScaler

# ==============================
#   配置路径
# ==============================
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
model_dir = os.path.join(root_dir, 'train/models')
data_dir = os.path.join(script_dir, 'data')
image_dir = os.path.join(data_dir, 'images')
mask_dir = os.path.join(data_dir, 'masks')
output_csv = os.path.join(script_dir, 'predictions.csv')

# ==============================
#   特征提取函数
# ==============================
def extract_features(image_path, mask_path):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None

    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    total_pixels = img.shape[0] * img.shape[1]
    bug_pixels = np.sum(mask_bin > 0)
    if bug_pixels == 0:
        return None

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

    area_ratio = bug_pixels / total_pixels

    return [
        area_ratio,
        R_mean, G_mean, B_mean, R_std, G_std, B_std,
        R_min, G_min, B_min, R_max, G_max, B_max,
        aspect_ratio, circularity, roughness, symmetry_score,
        *hu_moments_log
    ]
# ==============================
#   命令行参数
# ==============================
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, choices=[
    'random_forest', 'svm', 'knn', 'logistic_regression', 'all'
], help='Choose model from: random_forest, svm, knn, logistic_regression, all')
args = parser.parse_args()

# ==============================
#   批量读取图像和提取特征
# ==============================
results = []
for fname in sorted(os.listdir(image_dir)):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    img_id = os.path.splitext(fname)[0]
    image_path = os.path.join(image_dir, fname)
    mask_path = os.path.join(mask_dir, f"binary_{img_id}.tif")
    if not os.path.exists(mask_path):
        print(f"[Warning] Missing mask: {mask_path}")
        continue

    feats = extract_features(image_path, mask_path)
    if feats is None:
        continue

    results.append((fname, feats))

if not results:
    print("❌ 没有可用的图像/特征，无法预测。")
    exit()

X_test = np.array([r[1] for r in results])
fname_list = [r[0] for r in results]
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# ==============================
#   预测并保存 CSV
# ==============================
if args.model != 'all':
    model_path = os.path.join(model_dir, f"model_{args.model}.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)
    y_pred = model.predict(X_test_scaled)
    df = pd.DataFrame({"filename": fname_list, "predicted_label": y_pred})
    df.to_csv(output_csv, index=False)
    print(f"✅ 所有图像预测完成，结果保存在：{output_csv}")
else:
    all_models = ['random_forest', 'svm', 'knn', 'logistic_regression']
    all_dfs = []
    for name in all_models:
        path = os.path.join(model_dir, f"model_{name}.pkl")
        if not os.path.exists(path):
            print(f"[Skip] 模型不存在：{name}")
            continue
        model = joblib.load(path)
        try:
            if hasattr(model, 'n_features_in_') and model.n_features_in_ != X_test_scaled.shape[1]:
                print(f"[Skip] 特征维度不匹配：{name}")
                continue
            preds = model.predict(X_test_scaled)
        except ValueError:
            print(f"[Skip] 特征维度错误：{name}")
            continue
        df = pd.DataFrame({"filename": fname_list, f"pred_{name}": preds})
        all_dfs.append(df)
    if all_dfs:
        final_df = all_dfs[0]
        for df in all_dfs[1:]:
            final_df = final_df.merge(df, on="filename")
        final_df.to_csv(output_csv, index=False)
        print(f"✅ 所有模型预测完成，结果保存在：{output_csv}")
    else:
        print("❌ 所有模型都因特征维度不匹配被跳过，未生成预测结果。")