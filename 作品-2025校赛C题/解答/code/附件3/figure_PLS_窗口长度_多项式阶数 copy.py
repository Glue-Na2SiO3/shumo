import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.signal import savgol_filter
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']

# 窗口长度: 5, 多项式阶数: 4或窗口长度: 3, 多项式阶数: 2, 5折交叉验证平均准确率: 0.9083，最好

def read_data(file_path):
    df = pd.read_excel(file_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:]
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
    X = X.values.astype(float)
    # SG滤波
    X = np.array([savgol_filter(row, window_length=11, polyorder=2) for row in X])
    return X, y

def train_plsda(X, y, n_components=13):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, y)
    return scaler, pls

def predict_and_save_plsda(model, test_file, output_file):
    scaler, pls = model
    df_test = pd.read_excel(test_file)
    X_test = df_test.iloc[:, 1:].values.astype(float)
    # SG滤波
    X_test = np.array([savgol_filter(row, window_length=15, polyorder=2) for row in X_test])
    X_test_scaled = scaler.transform(X_test)
    pred = pls.predict(X_test_scaled)
    pred_label = np.round(pred).astype(int).flatten()
    df_result = pd.DataFrame({
        '样本ID': df_test.iloc[:, 0],
        '预测标签': pred_label
    })
    df_result.to_excel(output_file, index=False)

if __name__ == "__main__":
    window_lengths = [3,5,7, 9, 11, 13, 15]
    polyorders = [2, 3, 4,5]
    best_acc = 0
    best_params = None
    heatmap_data = []

    for window_length in window_lengths:
        row = []
        for polyorder in polyorders:
            if polyorder >= window_length:
                row.append(None)
                continue
            # 每次循环都重新读取原始数据
            df = pd.read_excel("总表.xlsx")
            y_train = df.iloc[:, 0].values
            X = df.iloc[:, 1:]
            X = X.apply(pd.to_numeric, errors='coerce')
            X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
            X = X.values.astype(float)
            # SG滤波
            X_train = np.array([savgol_filter(row, window_length=window_length, polyorder=polyorder) for row in X])
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            accs = []
            for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                scaler, pls = train_plsda(X_tr, y_tr, n_components=13)
                X_val_scaled = scaler.transform(X_val)
                pred = pls.predict(X_val_scaled)
                pred_label = np.round(pred).astype(int).flatten()
                acc = accuracy_score(y_val, pred_label)
                accs.append(acc)
                print(f"窗口长度: {window_length}, 多项式阶数: {polyorder}, 第{i+1}折准确率: {acc:.4f}")
            mean_acc = np.mean(accs)
            row.append(mean_acc)
            print(f"窗口长度: {window_length}, 多项式阶数: {polyorder}, 5折交叉验证平均准确率: {mean_acc:.4f}\n")
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_params = (window_length, polyorder)
        heatmap_data.append(row)

    # 将 None 替换为 np.nan
    heatmap_data = np.array(heatmap_data, dtype=object)
    heatmap_data = np.where(heatmap_data == None, np.nan, heatmap_data).astype(float)

    print(f"最优参数: 窗口长度={best_params[0]}, 多项式阶数={best_params[1]}, 平均准确率={best_acc:.4f}")

    # 绘制热力图
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", xticklabels=polyorders, yticklabels=window_lengths, cmap="coolwarm", cbar_kws={'label': '平均准确率'})
    plt.title("5折交叉验证平均准确率热力图", fontsize=16)
    plt.xlabel("多项式阶数", fontsize=12)
    plt.ylabel("窗口长度", fontsize=12)

    plt.tight_layout()
    # plt.savefig("heatmap.png")
    plt.show()