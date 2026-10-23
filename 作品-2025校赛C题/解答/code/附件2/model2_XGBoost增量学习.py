# XGBoost流式处理
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 数据加载
train_df = pd.read_excel('附件2处理/总表.xlsx')
test_df = pd.read_excel('附件2处理/问题2测试集.xlsx')

X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values

# 预处理
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)

# 参数搜索
param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.05]
}
model = XGBClassifier(tree_method='hist', enable_categorical=True)
search = RandomizedSearchCV(model, param_dist, n_iter=5, cv=3, n_jobs=-1)
search.fit(X_train, y_encoded)

# 增量训练
best_model = search.best_estimator_
best_model.fit(X_train, y_encoded, xgb_model=best_model)

# 交叉验证准确率
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_train_pred = cross_val_predict(best_model, X_train, y_encoded, cv=cv)
cv_acc = accuracy_score(y_encoded, y_train_pred)
print(f"交叉验证准确率: {cv_acc:.4f}")

# 结果输出
y_test_pred = best_model.predict(X_test)
result_df = pd.DataFrame({
    '样本编号': test_df.iloc[:, 0],
    '预测类别': le.inverse_transform(y_test_pred)
})
result_df.to_excel('问题2_XGBoost结果.xlsx', index=False)

# 特征重要性
from xgboost import plot_importance
import matplotlib.pyplot as plt

plot_importance(best_model, max_num_features=20)
# plt.show()