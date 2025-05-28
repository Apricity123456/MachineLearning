import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
import umap.umap_ as umap  # 安装：pip install umap-learn

# === 路径自动适配当前脚本目录 ===
base_dir = os.path.dirname(os.path.abspath(__file__))
features_path = os.path.join(base_dir, 'features_with_shape.csv')
labels_path = os.path.join(base_dir, 'classif.xlsx')
output_dir = os.path.join(base_dir, 'plots')
os.makedirs(output_dir, exist_ok=True)

# === 第一步：读取并合并特征和标签 ===
features_df = pd.read_csv(features_path)
labels_df = pd.read_excel(labels_path)

# 清理 ID 与标签列
features_df['ID'] = features_df['ID'].astype(str).str.extract(r"(\d+)")[0]
labels_df['ID'] = labels_df['ID'].astype(str).str.extract(r"(\d+)")[0]
labels_df = labels_df.rename(columns={'bug type': 'bug_type'})

# 合并
df = pd.merge(features_df, labels_df[['ID', 'bug_type']], on='ID', how='inner')

# === 第二步：特征 & 标签 ===
feature_cols = [col for col in df.columns if col not in ['ID', 'bug_type']]
X = df[feature_cols]
y = df['bug_type']

# === 可视化函数（统一风格） ===
def plot_projection(X_2d, method_name, save_name):
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y, palette='Set2', s=60)
    plt.title(f"{method_name} Projection of Insect Features")
    plt.xlabel(f"{method_name} 1")
    plt.ylabel(f"{method_name} 2")
    plt.legend(title="Bug Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    path = os.path.join(output_dir, save_name)
    plt.savefig(path)
    print(f"✅ {method_name} 图已保存：{path}")
    plt.show()

# === 1️⃣ PCA 二维可视化 ===
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X)
plot_projection(X_pca_2d, "PCA", "pca_2d.png")

# === 2️⃣ t-SNE（基于 20 维 PCA 特征）===
pca_20 = PCA(n_components=20, random_state=42)
X_pca_20 = pca_20.fit_transform(X)
tsne = TSNE(n_components=2, perplexity=20, n_iter=500, init='pca', learning_rate='auto', random_state=42)
X_tsne = tsne.fit_transform(X_pca_20)
plot_projection(X_tsne, "t-SNE", "tsne.png")

# === 3️⃣ UMAP ===
umap_2d = umap.UMAP(n_components=2, random_state=42)
X_umap = umap_2d.fit_transform(X)
plot_projection(X_umap, "UMAP", "umap.png")

# === 4️⃣ Isomap ===
isomap = Isomap(n_components=2)
X_isomap = isomap.fit_transform(X)
plot_projection(X_isomap, "Isomap", "isomap.png")
