# -*- coding: utf-8 -*-
"""
基于核PLS-DA+XGBoost的非线性光谱分类模型
1. 数据读取与合并
2. 预处理（归一化、S-G滤波、小波去噪）
3. 非线性特征工程（二阶差分、小波能量、导数-积分比等）
4. 核PLS-DA（RBF核）提取潜变量
5. XGBoost集成分类
6. 测试集推理与结果输出
"""
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import pywt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics.pairwise import rbf_kernel
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import os

# ======================== 数据读取 ========================
def read_data(file_path):
    """
    读取数据文件，返回X, y, 波长。
    自动将特征列转为数值，非数字转为NaN，再用每列均值填充空值。
    """
    df = pd.read_excel(file_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:]
    # 强制转为数值，非数字转为NaN
    X = X.apply(pd.to_numeric, errors='coerce')
    # 用每列均值填充空值
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
    X = X.values.astype(float)
    wavelengths = np.array([float(col) for col in df.columns[1:]])
    print(f"[LOG] 读取数据 {file_path} 完成，样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    return X, y, wavelengths

# ======================== 预处理 ========================
def preprocess(X):
    """
    1. 样本级归一化
    2. Savitzky-Golay滤波
    3. 小波去噪
    """
    # 归一化
    X_norm = (X - X.min(axis=1, keepdims=True)) / (X.max(axis=1, keepdims=True) - X.min(axis=1, keepdims=True) + 1e-8)
    # S-G滤波
    X_sg = savgol_filter(X_norm, window_length=11, polyorder=2, axis=1)
    # 小波去噪
    def wavelet_denoise(signal):
        coeffs = pywt.wavedec(signal, 'sym5', level=pywt.dwt_max_level(len(signal), 'sym5'))
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2*np.log(len(signal)))
        coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
        return pywt.waverec(coeffs, 'sym5')[:len(signal)]
    X_denoised = np.apply_along_axis(wavelet_denoise, 1, X_sg)
    print(f"[LOG] 预处理完成，形状: {X_denoised.shape}")
    return X_denoised

# ======================== 非线性特征工程 ========================
def extract_nonlinear_features(X, wavelengths):
    """
    1. 二阶差分
    2. 局部方差（5nm窗口）
    3. 小波能量（L1范数）
    4. 导数-积分比（600-800nm区间）
       """
    # 一阶/二阶差分
    diff1 = np.diff(X, axis=1)
    diff2 = np.diff(diff1, axis=1)
    # 局部方差（窗口10）
    window_size = 10
    # local_var = np.array([
    #     np.convolve(x, np.ones(window_size)/window_size, mode='valid') for x in X
    # ])
    local_var = np.array([
        [np.var(x[i:i+window_size]) for i in range(len(x)-window_size+1)] for x in X
    ])
    # 小波能量（L1范数）
    def get_wavelet_energy(signal):
        coeffs = pywt.wavedec(signal, 'sym5', level=3)
        return np.array([np.linalg.norm(c, ord=1) for c in coeffs])
    wavelet_energy = np.apply_along_axis(get_wavelet_energy, 1, X)
    # 导数-积分比（600-800nm区间）
    idx_600 = np.argmin(np.abs(wavelengths - 600))
    idx_800 = np.argmin(np.abs(wavelengths - 800))
    integral = X[:, idx_600:idx_800+1].sum(axis=1)
    deriv_mean = diff1[:, idx_600:idx_800].mean(axis=1)
    deriv_integral_ratio = integral / (deriv_mean + 1e-8)
    # 合并所有特征
    features = np.hstack([
        diff2,  # 二阶差分
        local_var,  # 局部方差
        wavelet_energy,  # 小波能量
        deriv_integral_ratio.reshape(-1,1)  # 导数-积分比
    ])
    print(f"[LOG] 特征工程完成，输出特征维数: {features.shape[1]}")
    return features

# ======================== 核PLS-DA + XGBoost训练 ========================
def train_kplsda_xgb(X, y, gamma=0.01, n_components=10, xgb_params=None):
    """
    1. RBF核映射
    2. PLSRegression提取潜变量
    3. XGBoost分类
    """
    # RBF核映射
    K_train = rbf_kernel(X, gamma=gamma)
    # PLS-DA
    pls = PLSRegression(n_components=n_components)
    T_train = pls.fit_transform(K_train, y)[0]
    # 合并特征
    X_combined = np.hstack([T_train, X])
    # XGBoost
    if xgb_params is None:
        xgb_params = dict(max_depth=3, n_estimators=100, learning_rate=0.1, subsample=0.8, reg_lambda=1)
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_combined, y)
    print(f"[LOG] 模型训练完成，PLS主成分: {n_components}, XGBoost参数: {xgb_params if xgb_params else '默认'}")
    return pls, xgb, X, gamma  # X为核参考集

# ======================== 预测与保存 ========================
def predict_and_save(model, test_file, output_file, wavelengths):
    """
    测试集推理与结果保存
    """
    pls, xgb, X_train_ref, gamma = model
    # 读取测试集
    df_test = pd.read_excel(test_file)
    X_test = df_test.iloc[:, 1:].values
    # 预处理
    X_test_pre = preprocess(X_test)
    # 特征工程
    test_features = extract_nonlinear_features(X_test_pre, wavelengths)
    # 核映射（测试集对训练集）
    K_test = rbf_kernel(test_features, X_train_ref, gamma=gamma)
    # PLS潜变量
    T_test = pls.transform(K_test)
    # 合并特征
    X_test_combined = np.hstack([T_test, test_features])
    # 预测
    pred = xgb.predict(X_test_combined)
    # 保存
    df_result = pd.DataFrame({
        '样本ID': df_test.iloc[:, 0],
        '预测标签': pred
    })
    df_result.to_excel(output_file, index=False)
    print(f"[LOG] 测试集 {test_file} 预测完成，结果已保存到 {output_file}")

# ======================== 主流程 ========================
if __name__ == "__main__":
    # 读取并合并训练集
    X1, y1, wavelengths = read_data("不同批次异同分析与通用模型训练集/第一批次数据训练集/第一批次总表.xlsx")
    X2, y2, _ = read_data("不同批次异同分析与通用模型训练集/第二批次数据训练集/第二批次总表.xlsx")
    X_train = np.vstack([X1, X2])
    y_train = np.concatenate([y1, y2])
    print("[LOG] 训练集合并完成")
    print(f"[LOG] 训练集X shape: {X_train.shape}, y shape: {y_train.shape}")
    print(f"[LOG] 训练集前5行X:", X_train[:5])
    print(f"[LOG] 训练集前5个y:", y_train[:5])
    print(f"[LOG] 训练集标签类别分布: {dict(pd.Series(y_train).value_counts().sort_index())}")
    # 预处理
    X_train_pre = preprocess(X_train)
    # 特征工程各部分
    diff1 = np.diff(X_train_pre, axis=1)
    diff2 = np.diff(diff1, axis=1)
    print(f"[LOG] diff2 shape: {diff2.shape}")
    window_size = 10
    local_var = np.array([
        np.convolve(x, np.ones(window_size)/window_size, mode='valid') for x in X_train_pre
    ])
    print(f"[LOG] local_var shape: {local_var.shape}")
    def get_wavelet_energy(signal):
        coeffs = pywt.wavedec(signal, 'sym5', level=3)
        return np.array([np.linalg.norm(c, ord=1) for c in coeffs])
    wavelet_energy = np.apply_along_axis(get_wavelet_energy, 1, X_train_pre)
    print(f"[LOG] wavelet_energy shape: {wavelet_energy.shape}")
    # 训练模型和测试集预测仅用diff2+local_var+wavelet_energy
    train_features = np.hstack([diff2, local_var, wavelet_energy])
    print(f"[LOG] 最终特征组合 shape: {train_features.shape}")
    print(f"[LOG] 训练集前5行特征均值: {np.mean(train_features[:5], axis=0)[:5]}")
    print(f"[LOG] 训练集前5行特征方差: {np.var(train_features[:5], axis=0)[:5]}")
    # 5折交叉验证准确率
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for i, (train_idx, val_idx) in enumerate(skf.split(train_features, y_train)):
        X_tr, X_val = train_features[train_idx], train_features[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model_cv = train_kplsda_xgb(X_tr, y_tr, gamma=0.01, n_components=10)
        pls, xgb, X_ref, gamma = model_cv
        K_val = rbf_kernel(X_val, X_ref, gamma=gamma)
        T_val = pls.transform(K_val)
        X_val_combined = np.hstack([T_val, X_val])
        y_pred = xgb.predict(X_val_combined)
        acc = accuracy_score(y_val, y_pred)
        accs.append(acc)
        print(f"[LOG] 第{i+1}折交叉验证准确率: {acc:.4f}")
    print(f"[LOG] 最终特征组合(diff2+local_var+wavelet_energy) 5折交叉验证平均准确率: {np.mean(accs):.4f}")
    model = train_kplsda_xgb(train_features, y_train, gamma=0.01, n_components=10)
    # 测试集预测与保存
    def extract_selected_features(X, wavelengths):
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
        feats = np.hstack([diff2, local_var, wavelet_energy])
        print(f"[LOG] 测试集特征 shape: {feats.shape}")
        return feats
    def predict_and_save_selected(model, test_file, output_file, wavelengths):
        pls, xgb, X_train_ref, gamma = model
        df_test = pd.read_excel(test_file)
        X_test = df_test.iloc[:, 1:].values
        print(f"[LOG] 测试集 {test_file} 样本数: {X_test.shape[0]}, 原始特征数: {X_test.shape[1]}")
        X_test_pre = preprocess(X_test)
        test_features = extract_selected_features(X_test_pre, wavelengths)
        K_test = rbf_kernel(test_features, X_train_ref, gamma=gamma)
        T_test = pls.transform(K_test)
        X_test_combined = np.hstack([T_test, test_features])
        print(f"[LOG] 测试集最终输入特征 shape: {X_test_combined.shape}")
        pred = xgb.predict(X_test_combined)
        df_result = pd.DataFrame({
            '样本ID': df_test.iloc[:, 0],
            '预测标签': pred
        })
        df_result.to_excel(output_file, index=False)
        print(f"[LOG] 测试集 {test_file} 预测完成，结果已保存到 {output_file}")
    test_files = [
        ("通用模型测试集/三分类测试集1.xlsx", "问题4测试集1结果.xlsx"),
        ("通用模型测试集/三分类测试集2.xlsx", "问题4测试集2结果.xlsx")
    ]
    for test_path, save_path in test_files:
        predict_and_save_selected(model, test_path, save_path, wavelengths)
    print("[LOG] 所有测试集预测完成")