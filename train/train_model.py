import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# === 自动获取路径 ===
base_dir = os.path.dirname(os.path.abspath(__file__))
features_path = os.path.join(base_dir, "features_with_shape.csv")
labels_path = os.path.join(base_dir, "classif.xlsx")
models_dir = os.path.join(base_dir, "models")

# 创建模型保存目录（如果不存在）
os.makedirs(models_dir, exist_ok=True)

# === 加载数据 ===
features_df = pd.read_csv(features_path)
labels_df = pd.read_excel(labels_path)

# === 处理 ID 格式 ===
features_df['ID'] = features_df['ID'].astype(str).str.extract(r'(\d+)')[0].astype(int)
labels_df['ID'] = labels_df['ID'].astype(str).str.extract(r'(\d+)')[0].astype(int)
labels_df = labels_df.rename(columns={"bug type": "bug_type"})

# === 合并数据 & 清洗类别 ===
df = pd.merge(features_df, labels_df[['ID', 'bug_type']], on='ID')
counts = df['bug_type'].value_counts()
df = df[df['bug_type'].isin(counts[counts >= 2].index)]

# === 特征与标签拆分 ===
X = df.drop(columns=["ID", "bug_type"])
y = df["bug_type"]

# === 特征标准化（重要！为逻辑回归和SVM提升性能）===
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === 拆分训练集 ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === 模型集合（监督学习）===
models = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "svm": SVC(kernel='rbf', probability=True, random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=3),
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42)
}

# === 打印交叉验证得分 & 保存模型 ===
print("📊 监督学习模型交叉验证准确率：")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name.capitalize()}: 准确率均值={scores.mean():.3f}，标准差={scores.std():.3f}")
    
    model.fit(X_train, y_train)
    model_path = os.path.join(models_dir, f"model_{name}.pkl")
    joblib.dump(model, model_path)
    print(f"✅ 模型已保存为: {model_path}")

# === 非监督学习：KMeans 聚类 ===
print("\n🔍 非监督学习：KMeans 聚类")
kmeans = KMeans(n_clusters=len(y.unique()), n_init=10, random_state=42)
clusters = kmeans.fit_predict(X)
df['cluster'] = clusters
print(df[['ID', 'bug_type', 'cluster']].head())
