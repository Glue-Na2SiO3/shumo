# -*- coding: utf-8 -*-
"""
基于PLS-DA的非线性光谱分类模型（仅PLS-DA，无XGBoost）
1. 数据读取与合并
2. 预处理（归一化、S-G滤波、小波去噪）
3. 非线性特征工程（二阶差分、局部方差、小波能量）
4. PLS-DA提取潜变量
5. 直接round分类
6. 输出训练集准确率
"""
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import pywt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ======================== 数据读取 ========================
def read_data(file_path):
    df = pd.read_excel(file_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:]
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
    X = X.values.astype(float)
    wavelengths = np.array([float(col) for col in df.columns[1:]])
    print(f"[LOG] 读取数据 {file_path} 完成，样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    return X, y, wavelengths

# ======================== 预处理 ========================
def preprocess(X):
    X_norm = (X - X.min(axis=1, keepdims=True)) / (X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True) + 1e-8)
    X_sg = savgol_filter(X_norm, window_length=11, polyorder=2, axis=1)
    def wavelet_denoise(signal):
        coeffs = pywt.wavedec(signal, 'sym5', level=pywt.dwt_max_level(len(signal), 'sym5'))
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2*np.log(len(signal)))
        coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
        return pywt.waverec(coeffs, 'sym5')[:len(signal)]
    X_denoised = np.apply_along_axis(wavelet_denoise, 1, X_sg)
    print(f"[LOG] 预处理完成，形状: {X_denoised.shape}")
    return X_denoised

# ======================== 特征工程 ========================
def extract_features(X):
    diff1 = np.diff(X, axis=1)
    diff2 = np.diff(diff1, axis=1)
    window_size = 10
    local_var = np.array([
        np.convolve(x, np.ones(window_size)/window_size, mode='valid') for x in X
    ])
    def get_wavelet_energy(signal):
        coeffs = pywt.wavedec(signal, 'sym5', level=3)
        return np.array([np.linalg.norm(c, ord=1) for c in coeffs])
    wavelet_energy = np.apply_along_axis(get_wavelet_energy, 1, X)
    features = np.hstack([diff2, local_var, wavelet_energy])
    print(f"[LOG] 特征工程完成，输出特征维数: {features.shape[1]}")
    return features

# ======================== PLS-DA训练 ========================
def train_plsda_cv(X, y, n_components=10, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_list = []
    fold = 1
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        pls = PLSRegression(n_components=n_components)
        pls.fit(X_tr, y_tr)
        y_pred = np.round(pls.predict(X_te).ravel())
        acc = accuracy_score(y_te, y_pred)
        acc_list.append(acc)
        print(f"[LOG] 第{fold}折准确率: {acc:.4f}")
        fold += 1
    print(f"[LOG] 5折平均准确率: {np.mean(acc_list):.4f}")
    return acc_list

# ======================== 主流程 ========================
if __name__ == "__main__":
    # 读取并合并训练集
    X1, y1, wavelengths = read_data("不同批次异同分析与通用模型训练集/第一批次数据训练集/第一批次总表.xlsx")
    X2, y2, _ = read_data("不同批次异同分析与通用模型训练集/第二批次数据训练集/第二批次总表.xlsx")
    X_train = np.vstack([X1, X2])
    y_train = np.concatenate([y1, y2])
    print("[LOG] 训练集合并完成")
    print(f"[LOG] 训练集X shape: {X_train.shape}, y shape: {y_train.shape}")
    # 预处理
    X_train_pre = preprocess(X_train)
    # 特征工程
    train_features = extract_features(X_train_pre)
    print(f"[LOG] 最终特征 shape: {train_features.shape}")
    # PLS-DA 5折交叉验证准确率
    acc_list = train_plsda_cv(train_features, y_train, n_components=10, n_splits=5)
    print("[LOG] 训练流程结束")

    # ====== 绘制PLS降维后三主成分三维分布 ======
    pls = PLSRegression(n_components=3)
    T = pls.fit_transform(train_features, y_train)[0]  # T shape: (n_samples, 3)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 定义颜色映射（与描述中的颜色一致）
    colors = ['green', 'orange', 'blue']
    labels = ['类别0', '类别1', '类别2']
    
    # 分别绘制每一类别的点
    for i in range(3):
        mask = (y_train == i)
        ax.scatter(T[mask, 0], T[mask, 1], T[mask, 2], 
                   c=colors[i], label=labels[i], s=30, alpha=0.8)
    
    # 设置坐标轴标签和范围（与描述中的范围一致）
    ax.set_xlabel('PLS主成分1', fontsize=12, labelpad=10)
    ax.set_ylabel('PLS主成分2', fontsize=12, labelpad=10)
    ax.set_zlabel('PLS主成分3', fontsize=12, labelpad=10)
    
    ax.set_xlim(-80, 50)
    ax.set_ylim(-60, 15)
    ax.set_zlim(-30, 50)
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=10, title='类别')
    
    # 设置标题
    plt.title('PLS降维后前三主成分三维分布', fontsize=14, pad=20)
    
    # 调整视角
    ax.view_init(elev=25, azim=45)
    
    # 显示网格
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()