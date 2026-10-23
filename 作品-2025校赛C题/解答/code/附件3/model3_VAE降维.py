# 变分自编码器
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from scipy.signal import savgol_filter

def preprocess_data(df):
    X = df.iloc[:, 1:]
    X = X.apply(pd.to_numeric, errors='coerce')
    # clip到[0, 100]区间
    X = X.clip(lower=0, upper=100)
    # 对每一列用线性插值填补clip后的缺失值
    X = X.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')
    X = X.values.astype(float)
    # SG滤波
    X = np.array([savgol_filter(row, window_length=3, polyorder=2) for row in X])
    return X

def train_vae(X_train, y_train, model_path='vae_model.pkl', cv=5):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    latent_dim = 64
    encoder = tf.keras.Sequential([
        layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Dense(latent_dim)
    ])
    X_train_encoded = encoder.predict(X_train)
    classifier = LGBMClassifier(verbose=-1)
    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    acc_list = []
    f1_list = []
    for train_idx, test_idx in cv_split.split(X_train_encoded, y_encoded):
        X_tr, X_val = X_train_encoded[train_idx], X_train_encoded[test_idx]
        y_tr, y_val = y_encoded[train_idx], y_encoded[test_idx]
        classifier.fit(X_tr, y_tr)
        y_pred = classifier.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        acc_list.append(acc)
        f1_list.append(f1)
    mean_acc = float(np.mean(acc_list)) if len(acc_list) > 0 else 0.0
    mean_f1 = float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0
    classifier.fit(X_train_encoded, y_encoded)
    if model_path is not None:
        joblib.dump({'encoder': encoder, 'classifier': classifier, 'le': le, 'cv_acc_list': acc_list, 'cv_acc_mean': mean_acc, 'cv_f1_list': f1_list, 'cv_f1_mean': mean_f1}, model_path)
    return {'encoder': encoder, 'classifier': classifier, 'le': le, 'cv_acc_list': acc_list, 'cv_acc_mean': mean_acc, 'cv_f1_list': f1_list, 'cv_f1_mean': mean_f1, 'y_encoded': y_encoded, 'X_train_encoded': X_train_encoded}

def predict_vae(X_test, model_path='vae_model.pkl'):
    data = joblib.load(model_path)
    encoder = data['encoder']
    classifier = data['classifier']
    le = data['le']
    X_test_encoded = encoder.predict(X_test)
    y_test_pred = classifier.predict(X_test_encoded)
    y_test_labels = le.inverse_transform(y_test_pred)
    return y_test_labels

def main_train():
    train_df = pd.read_excel('总表.xlsx')
    X_train = preprocess_data(train_df)
    y_train = train_df.iloc[:, 0].values
    model_data = train_vae(X_train, y_train)
    print(f"5折交叉验证准确率: {model_data['cv_acc_list']}")
    print(f"平均交叉验证准确率: {model_data['cv_acc_mean']:.4f}")
    print(f"5折交叉验证F1: {model_data['cv_f1_list']}")
    print(f"平均交叉验证F1: {model_data['cv_f1_mean']:.4f}")
    train_pred = model_data['classifier'].predict(model_data['X_train_encoded'])
    ConfusionMatrixDisplay.from_estimator(model_data['classifier'], model_data['X_train_encoded'], model_data['y_encoded'])
    plt.title('Confusion Matrix')
    plt.show()
    train_acc = accuracy_score(model_data['y_encoded'], train_pred)
    print(f"训练集准确率: {train_acc:.4f}")

def main_test():
    test_df = pd.read_excel('测试集数据.xlsx')
    X_test = preprocess_data(test_df)
    y_test_labels = predict_vae(X_test)
    result_df = pd.DataFrame({
        '样本编号': test_df.iloc[:, 0],
        '预测类别': y_test_labels
    })
    result_df.to_excel('问题2_VAE结果.xlsx', index=False)
    print('测试集预测结果已保存。')

if __name__ == '__main__':
    import os
    if not os.path.exists('vae_model.pkl'):
        main_train()
    main_test()