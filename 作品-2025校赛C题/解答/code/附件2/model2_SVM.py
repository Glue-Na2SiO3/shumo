import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ========== 数据读取 ==========
train_df = pd.read_excel('附件2处理/总表.xlsx')
test_df = pd.read_excel('附件2处理/问题2测试集.xlsx')

# ========== 数据预处理 ==========
# 提取特征和标签
X_train = train_df.iloc[:, 1:].values  # 假设第一列为标签
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values

# 标签编码
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# 特征选择
selector = VarianceThreshold(threshold=0.1)
X_train_selected = selector.fit_transform(X_train)
X_test_selected = selector.transform(X_test)

# ========== 参数网格搜索 ==========
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.1],
    'kernel': ['rbf']
}

# ========== 模型训练 ==========
svm = SVC(probability=True)
grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train_selected, y_train_encoded)

# 输出最佳参数
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV Accuracy: {grid_search.best_score_:.2%}")

# ========== 交叉验证评估 ==========
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_train_pred = cross_val_predict(
    grid_search.best_estimator_, 
    X_train_selected, 
    y_train_encoded, 
    cv=cv
)
cv_acc = accuracy_score(y_train_encoded, y_train_pred)
print(f"交叉验证准确率: {cv_acc:.4f}")

# ========== 测试集预测 ==========
best_model = grid_search.best_estimator_

# 计算训练集准确率
train_acc = accuracy_score(y_train_encoded, best_model.predict(X_train_selected))
print(f"训练集准确率: {train_acc:.4f}")

y_test_pred = best_model.predict(X_test_selected)
y_test_labels = le.inverse_transform(y_test_pred)

# ========== 结果保存 ==========
result_df = pd.DataFrame({
    '样本编号': test_df.iloc[:, 0],  # 假设第一列为样本ID
    '预测类别': y_test_labels
})
result_df.to_excel('问题2_SVM结果.xlsx', index=False)

# ========== 可视化分析 ==========
# 混淆矩阵
fig, ax = plt.subplots(1, 2, figsize=(15,6))

# 训练集混淆矩阵
ConfusionMatrixDisplay.from_estimator(
    best_model, 
    X_train_selected, 
    y_train_encoded,
    display_labels=le.classes_,
    ax=ax[0],
    cmap='Blues'
)
ax[0].set_title('Training Set Confusion Matrix')

# 特征重要性（SVM权重）
if best_model.kernel == 'linear':
    feature_importance = np.abs(best_model.coef_).sum(axis=0)
    ax[1].bar(range(len(feature_importance)), feature_importance)
    ax[1].set_title('Feature Importance (SVM Coefficients)')
    ax[1].set_xlabel('Feature Index')
    ax[1].set_ylabel('Importance')

plt.tight_layout()
plt.savefig('SVM_analysis.png', dpi=300)
# plt.show()

# ========== 关键波长分析 ==========
if best_model.kernel == 'linear':
    top_features = np.argsort(feature_importance)[-10:][::-1]
    print("Top 10 Important Features Indices:", top_features)
    
    # 假设波长列为400-1000nm，间隔5nm（需根据实际数据调整）
    wavelengths = np.linspace(400, 1000, X_train.shape[1])
    print("Corresponding Wavelengths (nm):", wavelengths[top_features])