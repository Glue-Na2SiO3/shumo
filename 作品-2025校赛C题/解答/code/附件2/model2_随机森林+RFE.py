# 随机森林递归特征消除
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 数据加载
train_df = pd.read_excel('附件2处理/总表.xlsx')
test_df = pd.read_excel('附件2处理/问题2测试集.xlsx')

X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values

# 标签编码
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)

# 特征选择
estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
selector = RFECV(estimator, step=25, cv=5, min_features_to_select=50, n_jobs=-1)  # 调整步长和最小特征数
X_train_selected = selector.fit_transform(X_train, y_encoded)
X_test_selected = selector.transform(X_test)
print(f"递归特征消除后特征数: {X_train_selected.shape[1]}")

# 优化模型
model = RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1)
model.fit(X_train_selected, y_encoded)

# 交叉验证准确率
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_train_pred = cross_val_predict(model, X_train_selected, y_encoded, cv=cv)
cv_acc = accuracy_score(y_encoded, y_train_pred)
print(f"交叉验证准确率: {cv_acc:.4f}")

# 测试集预测
y_test_pred = model.predict(X_test_selected)
result_df = pd.DataFrame({
    '样本编号': test_df.iloc[:, 0],
    '预测类别': le.inverse_transform(y_test_pred)
})
result_df.to_excel('问题2_RFE随机森林结果.xlsx', index=False)

# 可视化
fig, ax = plt.subplots(1, 2, figsize=(15,6))
ConfusionMatrixDisplay.from_estimator(model, X_train_selected, y_encoded, ax=ax[0])
ax[0].set_title('Confusion Matrix')

importances = model.feature_importances_
ax[1].bar(range(len(importances)), importances)
ax[1].set_title('Feature Importances')
# plt.show()