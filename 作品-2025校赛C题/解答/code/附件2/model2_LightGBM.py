import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import time

# ========== 数据读取 ==========
train_df = pd.read_excel('附件2处理/总表.xlsx')
test_df = pd.read_excel('附件2处理/问题2测试集.xlsx')

# ========== 数据预处理 ========== 
# 提取特征和标签
X_train = train_df.iloc[:, 1:].values  # 第1列起为特征
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values

# 标签编码
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# ====== 特征选择（两阶段降维）====== 
# 第一阶段：方差过滤（去除低方差特征）
var_thres = VarianceThreshold(threshold=0.1)  
X_train_var = var_thres.fit_transform(X_train)
print(f"方差过滤后特征数: {X_train_var.shape[1]}")

# 第二阶段：基于随机森林的重要性选择
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_var, y_train_encoded)

# 选择重要性前200的特征
selector = SelectFromModel(rf, prefit=True, max_features=200)
X_train_reduced = selector.transform(X_train_var)
selected_indices = selector.get_support(indices=True)
print(f"最终保留特征数: {X_train_reduced.shape[1]}")

# 应用相同变换到测试集
X_test_reduced = selector.transform(var_thres.transform(X_test))

# ========== 模型训练 ==========
# 使用LightGBM（高效处理高维数据）
params = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'random_state': 42,
    'n_jobs': -1
}


lgb = LGBMClassifier(**params, verbose=-1)  # 设置verbose=-1以禁用日志输出

# 交叉验证预测
start_time = time.time()
cv = StratifiedKFold(n_splits=5, shuffle=True)
y_pred = cross_val_predict(lgb, X_train_reduced, y_train_encoded, cv=cv)
cv_time = time.time() - start_time
print(f"交叉验证耗时: {cv_time:.2f}秒")

# 准确率评估
acc = accuracy_score(y_train_encoded, y_pred)
print(f"交叉验证准确率: {acc:.4f}")

# 全数据训练
lgb.fit(X_train_reduced, y_train_encoded)

# ========== 测试集预测 ==========
y_test_pred = lgb.predict(X_test_reduced)
y_test_labels = le.inverse_transform(y_test_pred)

# 保存结果
result_df = pd.DataFrame({
    '样本编号': test_df.iloc[:, 0],  # 假设第一列为样本ID
    '预测类别': y_test_labels
})
result_df.to_excel('问题2_LightGBM结果.xlsx', index=False)

# ========== 关键波长分析 ==========
# 获取最终使用的特征在原数据中的位置
original_indices = var_thres.get_support(indices=True)[selected_indices]
print("关键波长索引:", original_indices[:10])  # 输出前10重要特征

# 假设波长列为400-2500nm，间隔1nm（需根据实际数据调整）
wavelengths = np.arange(400, 400 + X_train.shape[1])
print("对应波长(nm):", wavelengths[original_indices[:10]])