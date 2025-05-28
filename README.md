# Bug Type Classification System

本项目是一个基于图像处理与机器学习的系统，用于自动识别和分类昆虫类型。我们使用带有掩码的图像提取颜色与形状特征，并应用多个监督与非监督学习模型进行训练与预测。

---

##  项目结构

```
ML/
├── train/
│   ├── data/
│   │   ├── images/         # 训练图像
│   │   └── masks/          # 与训练图像对应的掩码
│   ├── models/             # 保存的机器学习模型（.pkl）
│   ├── plots/              # 可视化输出（如降维图）
│   ├── feature_extraction.py  # 提取特征并保存为 CSV
│   ├── visualize_features.py  # 可视化特征分布（PCA+tSNE）
│   └── train_model.py         # 模型训练与保存脚本
├── test/
│   ├── data/
│   │   ├── images/         # 测试图像
│   │   └── masks/          # 与测试图像对应的掩码
│   ├── predict_all_images.py  # 预测脚本，支持多模型比较
│   └── predictions.csv         # 模型预测结果
└── README.md
```

---

##  安装依赖

建议使用 Python 3.10 或以上版本。

```bash
pip install -r requirements.txt
```

或手动安装主要依赖：

```bash
pip install pandas numpy scikit-learn opencv-python joblib
```

---

##  训练模型

进入 `train/` 目录，运行训练脚本：

```bash
python train_model.py
```

该脚本将：
- 提取训练图像和掩码的特征
- 使用 4 个监督学习模型训练：Random Forest、SVM、KNN、Logistic Regression
- 进行交叉验证并输出准确率
- 将每个模型保存为 `.pkl` 文件至 `train/models/` 文件夹

---

##  执行预测

在项目根目录下运行以下命令（注意切换路径）：

```bash
python test/predict_all_images.py --model all
```

可用参数有：
- `--model random_forest`
- `--model svm`
- `--model knn`
- `--model logistic_regression`
- `--model all`（对所有模型进行预测并汇总结果）

预测结果将保存为 CSV 文件：`test/predictions.csv`

---

## 输出示例（使用 `--model all`）

| filename        | pred_random_forest | pred_svm | pred_knn | pred_logistic_regression |
|-----------------|--------------------|----------|----------|---------------------------|
| image_001.png   | Bumblebee          | Bee      | Bee      | Bee                       |
| image_002.png   | Wasp               | Wasp     | Wasp     | Wasp                      |

---

##  特征说明

每张图像通过掩码提取以下 24 维特征：
- 颜色统计量（均值、标准差、最大/最小值）× 3 通道（R/G/B）
- 轮廓信息（面积、周长、圆度、粗糙度、长宽比）
- 对称性得分
- Hu不变矩（7维）

---

##  特征提取与可视化

### `train/feature_extraction.py`

该脚本从训练图像及其掩码中提取结构化数值特征，主要流程包括：

- 从图像中获取掩码覆盖区域（即昆虫区域）的像素；
- 统计颜色通道（RGB）的均值、标准差、最大值、最小值；
- 分析轮廓（面积、周长、长宽比、圆度、粗糙度）；
- 计算左右对称性得分；
- 提取 Hu 不变矩（用于刻画形状）。

执行该脚本后会生成一个包含所有图像特征的 CSV 文件：

```
features_with_shape.csv
```

---

### `train/visualize_features.py`

该脚本用于对提取到的高维特征数据进行降维并可视化，用于验证特征分布与不同昆虫类型的可分性。主要步骤：

1. 使用 `PCA` 将原始 24 维特征降维到 20 维；
2. 再通过 `t-SNE` 将其进一步降维至 2D 空间；
3. 用 `seaborn` 绘图展示各类昆虫在 2D 空间的聚类情况；
4. 输出图像保存为：

```
train/plots/tsne_plot.png
```

该图像可以清晰展示模型能否通过特征成功区分不同类型昆虫。

---
