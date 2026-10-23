# 主成分分析+逻辑回归
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 数据加载
train_df = pd.read_excel('附件2处理/总表.xlsx')
test_df = pd.read_excel('附件2处理/问题2测试集.xlsx')

X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values

# 预处理
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)

# PCA降维
pca = PCA(n_components=0.90)  # 保留90%方差
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 模型训练
model = LogisticRegressionCV(Cs=10, cv=5, max_iter=10000, multi_class='multinomial')
# 交叉验证准确率
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_train_pred = cross_val_predict(model, X_train_pca, y_encoded, cv=cv)
cv_acc = accuracy_score(y_encoded, y_train_pred)
print(f"交叉验证准确率: {cv_acc:.4f}")
model.fit(X_train_pca, y_encoded)

# 结果输出
y_test_pred = model.predict(X_test_pca)
result_df = pd.DataFrame({
    '样本编号': test_df.iloc[:, 0],
    '预测类别': le.inverse_transform(y_test_pred)
})
result_df.to_excel('问题2_PCA逻辑回归结果.xlsx', index=False)

# 方差解释率可视化
plt.figure(figsize=(10,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
# plt.show()