# 变分自编码器
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# 数据加载
train_df = pd.read_excel('附件2处理/总表.xlsx')
test_df = pd.read_excel('附件2处理/问题2测试集.xlsx')

X_train = train_df.iloc[:, 1:].values.astype(np.float32)
y_train = train_df.iloc[:, 0].values
X_test = test_df.iloc[:, 1:].values.astype(np.float32)

# 预处理
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)

# 构建VAE
latent_dim = 64
encoder = tf.keras.Sequential([
    layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    layers.Dense(latent_dim)
])

# 特征提取
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# 分类模型
classifier = LGBMClassifier(verbose=-1)

# 交叉验证准确率
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_train_pred = cross_val_predict(classifier, X_train_encoded, y_encoded, cv=cv)
cv_acc = accuracy_score(y_encoded, y_train_pred)
print(f"交叉验证准确率: {cv_acc:.4f}")

# 训练模型
classifier.fit(X_train_encoded, y_encoded)

# 计算训练集准确率
train_acc = accuracy_score(y_encoded, classifier.predict(X_train_encoded))
print(f"训练集准确率: {train_acc:.4f}")

# 预测
y_test_pred = classifier.predict(X_test_encoded)
result_df = pd.DataFrame({
    '样本编号': test_df.iloc[:, 0],
    '预测类别': le.inverse_transform(y_test_pred)
})
result_df.to_excel('问题2_VAE结果.xlsx', index=False)