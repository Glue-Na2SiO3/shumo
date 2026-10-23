# 堆叠集成学习
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# 数据加载
train_df = pd.read_excel('附件2处理/总表.xlsx')
test_df = pd.read_excel('附件2处理/问题2测试集.xlsx')

X_train = train_df.iloc[:, 1:].values
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values

# 预处理
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)

# 定义基学习器
base_learners = [
    ('svm', SVC(kernel='rbf', probability=True, C=1)),
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=5)),
    ('xgb', XGBClassifier(learning_rate=0.05, n_estimators=100))
]

# 构建堆叠模型
stacker = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(max_iter=10000),
    cv=5,
    n_jobs=-1
)

# 交叉验证准确率
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_train_pred = cross_val_predict(stacker, X_train, y_encoded, cv=cv)
cv_acc = accuracy_score(y_encoded, y_train_pred)
print(f"交叉验证准确率: {cv_acc:.4f}")

# 训练
stacker.fit(X_train, y_encoded)

# 预测
y_test_pred = stacker.predict(X_test)
result_df = pd.DataFrame({
    '样本编号': test_df.iloc[:, 0],
    '预测类别': le.inverse_transform(y_test_pred)
})
result_df.to_excel('问题2_Stacking结果.xlsx', index=False)