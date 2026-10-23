import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# 读取数据（假设数据文件为CSV格式）
train_data = pd.read_excel('附件2处理\总表.xlsx')
test_data = pd.read_excel('附件2处理\问题2测试集.xlsx')

# 数据预处理
X_train = train_data.iloc[:, 1:].values  # 去掉第一列类别标签
y_train = train_data.iloc[:, 0].values  # 第一列是类别标签
X_test = test_data.iloc[:, 1:].values  # 去掉第一列类别标签

# 标签编码
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 检查原始特征数量
print(f"原始特征数量: {X_train.shape[1]}")

# 检查标准化后特征数量
print(f"标准化后特征数量: {X_train_scaled.shape[1]}")

# 最佳成分数优化
best_acc = 0
best_n = 2
explained_var_pls = []
max_lv = 20
for n in range(1, max_lv + 1):
    pls = PLSRegression(n_components=n)
    pls.fit(X_train_scaled, y_encoded)
    X_scores = pls.transform(X_train_scaled)
    X_hat = np.dot(X_scores, pls.x_loadings_.T) + pls._x_mean
    var_ratio = 1 - np.sum((X_train_scaled - X_hat) ** 2) / np.sum((X_train_scaled - np.mean(X_train_scaled, axis=0)) ** 2)
    explained_var_pls.append(var_ratio)
    # 使用交叉验证预测
    y_pred = cross_val_predict(pls, X_train_scaled, y_encoded, cv=5)
    # 将连续值的预测结果转换为离散类别标签
    y_pred_rounded = np.round(y_pred).astype(int)
    # 计算准确率
    acc = accuracy_score(y_encoded, y_pred_rounded)
    if acc > best_acc:
        best_acc = acc
        best_n = n

# 训练最终模型
pls_final = PLSRegression(n_components=best_n)
pls_final.fit(X_train_scaled, y_encoded)

# 预测测试集
y_test_pred = pls_final.predict(X_test_scaled).round().astype(int)
y_test_pred = np.clip(y_test_pred, 0, len(le.classes_)-1).flatten()
y_test_labels = le.inverse_transform(y_test_pred)

# 保存结果
result_df = pd.DataFrame({'预测类别': y_test_labels})
result_df.to_excel('问题2_PLS-DA结果.xlsx', index=False)

# 绘制特征重要性图（如果适用）
if hasattr(pls_final, 'x_weights_'):
    feature_importance = np.sum(np.abs(pls_final.x_weights_), axis=1)
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.show()

# 仅PLS-DA累计解释方差曲线
plt.figure(figsize=(8, 5))
sns.set(style="ticks", font_scale=1.3)
plt.plot(range(1, max_lv+1), explained_var_pls, marker='o', color='#1f77b4', linewidth=2.5, label='PLS-DA Cumulative Explained Variance')
plt.axvline(best_n, color='#d62728', linestyle='--', linewidth=2, label=f'Optimal LV = {best_n}')
plt.scatter([best_n], [explained_var_pls[best_n-1]], color='#d62728', s=120, zorder=5)
plt.text(best_n+0.5, explained_var_pls[best_n-1], f'{explained_var_pls[best_n-1]:.2%}', color='#d62728', fontsize=14, va='center', fontweight='bold')
plt.xlabel('Number of Latent Variables', fontweight='bold')
plt.ylabel('Cumulative Explained Variance', fontweight='bold')
plt.title('PLS-DA Cumulative Explained Variance', fontweight='bold')
plt.ylim(0, 1.05)
plt.xlim(1, max_lv)
plt.legend(frameon=True)
sns.despine()
plt.tight_layout()
plt.savefig('PLSDA_累计解释方差曲线.png', dpi=300)
plt.close()

# 仅PLS-DA权重系数-波长曲线
if hasattr(pls_final, 'x_weights_'):
    plt.figure(figsize=(11, 5))
    sns.set(style="ticks", font_scale=1.3)
    weights = pls_final.x_weights_[:, 0]
    wave_idx = np.arange(1, len(weights)+1)
    plt.plot(wave_idx, weights, color='#2ca02c', linewidth=2.5, label='1st Component Weights')
    top_idx = np.argsort(np.abs(weights))[::-1][:5]
    plt.scatter(wave_idx[top_idx], weights[top_idx], color='#d62728', s=120, zorder=5, label='Top Bands')
    for idx in top_idx:
        plt.annotate(f'Band {wave_idx[idx]}',
                     xy=(wave_idx[idx], weights[idx]),
                     xytext=(wave_idx[idx]+10, weights[idx]+0.02*np.sign(weights[idx])),
                     arrowprops=dict(facecolor='#d62728', shrink=0.05, width=2, headwidth=8),
                     fontsize=12, color='#d62728', fontweight='bold')
    plt.xlabel('Band Index', fontweight='bold')
    plt.ylabel('Weight Coefficient', fontweight='bold')
    plt.title('PLS-DA 1st Component Weights vs. Wavelength', fontweight='bold')
    plt.legend(frameon=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig('PLSDA_权重系数波长曲线.png', dpi=300)
    plt.close()

# 计算混淆矩阵
train_pred = pls_final.predict(X_train_scaled).round().astype(int)
train_pred = np.clip(train_pred, 0, len(le.classes_)-1).flatten()
conf_matrix = confusion_matrix(y_encoded, train_pred)
ConfusionMatrixDisplay(conf_matrix, display_labels=le.classes_).plot()
plt.title('Confusion Matrix on Training Set')
plt.show()

# 输出训练集准确率
train_acc = accuracy_score(y_encoded, train_pred)
print(f"训练集准确率: {train_acc:.4f}")