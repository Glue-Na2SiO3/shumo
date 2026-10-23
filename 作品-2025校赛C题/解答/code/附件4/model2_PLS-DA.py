import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# 读取数据（假设数据文件为Excel格式，第一行为（类别，特征1，特征2...））
train_data = pd.read_excel('不同批次异同分析与通用模型训练集\两批次总表_combat.xlsx')
test_data = pd.read_excel('通用模型测试集\三分类测试集1处理.xlsx')

# 数据预处理
# 训练集：第一列为类别（0,1,2），其余为特征
X_train = train_data.iloc[:, 1:].values  # 特征
y_train = train_data.iloc[:, 0].values   # 类别标签（0,1,2）
# 测试集：全部为特征（不含类别标签）
X_test = test_data.iloc[:, 1:].values

# 标签编码（此处类别已为0,1,2，可省略，但保留以防万一）
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)

# OneHot编码用于多分类PLS-DA（兼容老版本scikit-learn，使用pandas实现）
y_onehot = pd.get_dummies(y_encoded).values

# 不进行标准化，直接使用原始特征
X_train_used = X_train
X_test_used = X_test

# 检查原始特征数量
print(f"原始特征数量: {X_train.shape[1]}")

# 可选：特征选择（如方差过滤、相关性分析等）
# 例如：去除方差为0的特征
# from sklearn.feature_selection import VarianceThreshold
# selector = VarianceThreshold(threshold=0)
# X_train_used = selector.fit_transform(X_train)
# X_test_used = selector.transform(X_test)

# 最佳成分数优化
best_acc = 0
best_n = 2
explained_var_pls = []
max_lv = 40  # 增大最大成分数
for n in range(1, max_lv + 1):
    pls = PLSRegression(n_components=n)
    pls.fit(X_train_used, y_onehot)
    # 使用交叉验证预测
    y_pred = cross_val_predict(pls, X_train_used, y_onehot, cv=10)
    y_pred_label = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_encoded, y_pred_label)
    if acc > best_acc:
        best_acc = acc
        best_n = n

# 训练最终模型
pls_final = PLSRegression(n_components=best_n)
pls_final.fit(X_train_used, y_onehot)

# 预测测试集
# 预测结果为0,1,2三类
proba_test = pls_final.predict(X_test_used)
y_test_pred = np.argmax(proba_test, axis=1)
y_test_labels = le.inverse_transform(y_test_pred)

# 保存结果
result_df = pd.DataFrame({'预测类别': y_test_labels})
result_df.to_excel('通用模型测试集/问题4_PLS-DA结果.xlsx', index=False)

# # 绘制特征重要性图（如果适用）
# if hasattr(pls_final, 'x_weights_'):
#     feature_importance = np.sum(np.abs(pls_final.x_weights_), axis=1)
#     plt.figure(figsize=(10, 6))
#     plt.bar(range(len(feature_importance)), feature_importance)
#     plt.title('Feature Importance')
#     plt.xlabel('Feature Index')
#     plt.ylabel('Importance')
#     plt.show()

# # 仅PLS-DA累计解释方差曲线
# plt.figure(figsize=(8, 5))
# sns.set(style="ticks", font_scale=1.3)
# plt.plot(range(1, max_lv+1), explained_var_pls, marker='o', color='#1f77b4', linewidth=2.5, label='PLS-DA Cumulative Explained Variance')
# plt.axvline(best_n, color='#d62728', linestyle='--', linewidth=2, label=f'Optimal LV = {best_n}')
# plt.scatter([best_n], [explained_var_pls[best_n-1]], color='#d62728', s=120, zorder=5)
# plt.text(best_n+0.5, explained_var_pls[best_n-1], f'{explained_var_pls[best_n-1]:.2%}', color='#d62728', fontsize=14, va='center', fontweight='bold')
# plt.xlabel('Number of Latent Variables', fontweight='bold')
# plt.ylabel('Cumulative Explained Variance', fontweight='bold')
# plt.title('PLS-DA Cumulative Explained Variance', fontweight='bold')
# plt.ylim(0, 1.05)
# plt.xlim(1, max_lv)
# plt.legend(frameon=True)
# sns.despine()
# plt.tight_layout()
# plt.savefig('PLSDA_累计解释方差曲线.png', dpi=300)
# plt.close()

# # 仅PLS-DA权重系数-波长曲线
# if hasattr(pls_final, 'x_weights_'):
#     plt.figure(figsize=(11, 5))
#     sns.set(style="ticks", font_scale=1.3)
#     weights = pls_final.x_weights_[:, 0]
#     wave_idx = np.arange(1, len(weights)+1)
#     plt.plot(wave_idx, weights, color='#2ca02c', linewidth=2.5, label='1st Component Weights')
#     top_idx = np.argsort(np.abs(weights))[::-1][:5]
#     plt.scatter(wave_idx[top_idx], weights[top_idx], color='#d62728', s=120, zorder=5, label='Top Bands')
#     for idx in top_idx:
#         plt.annotate(f'Band {wave_idx[idx]}',
#                      xy=(wave_idx[idx], weights[idx]),
#                      xytext=(wave_idx[idx]+10, weights[idx]+0.02*np.sign(weights[idx])),
#                      arrowprops=dict(facecolor='#d62728', shrink=0.05, width=2, headwidth=8),
#                      fontsize=12, color='#d62728', fontweight='bold')
#     plt.xlabel('Band Index', fontweight='bold')
#     plt.ylabel('Weight Coefficient', fontweight='bold')
#     plt.title('PLS-DA 1st Component Weights vs. Wavelength', fontweight='bold')
#     plt.legend(frameon=True)
#     sns.despine()
#     plt.tight_layout()
#     plt.savefig('PLSDA_权重系数波长曲线.png', dpi=300)
#     plt.close()

# # 计算混淆矩阵
train_pred = np.argmax(pls_final.predict(X_train_used), axis=1)
# conf_matrix = confusion_matrix(y_encoded, train_pred)
# ConfusionMatrixDisplay(conf_matrix, display_labels=le.classes_).plot()
# plt.title('Confusion Matrix on Training Set')
# plt.show()

# 输出训练集准确率
train_acc = accuracy_score(y_encoded, train_pred)
print(f"训练集准确率: {train_acc:.4f}")