import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.signal import savgol_filter
import openpyxl
import matplotlib.pyplot as plt
import os
plt.rcParams['font.sans-serif'] = ['SimHei']

# 13效果最好



def read_data(file_path):
    df = pd.read_excel(file_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:]
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.apply(lambda col: col.fillna(col.mean()), axis=0)
    X = X.values.astype(float)
    # SG滤波
    X = np.array([savgol_filter(row, window_length=3, polyorder=2) for row in X])
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
    X_test = np.array([savgol_filter(row, window_length=3, polyorder=2) for row in X_test])
    X_test_scaled = scaler.transform(X_test)
    pred = pls.predict(X_test_scaled)
    pred_label = np.round(pred).astype(int).flatten()
    df_result = pd.DataFrame({
        '样本ID': df_test.iloc[:, 0],
        '预测标签': pred_label
    })
    df_result.to_excel(output_file, index=False)

if __name__ == "__main__":
    # 最优参数
    window_length = 3
    polyorder = 2

    # n_components 循环测试
    result_file = "n_components_results.xlsx"
    if os.path.exists(result_file):
        df_result = pd.read_excel(result_file)
        n_components_range = df_result['n_components']
        mean_accuracies = df_result['mean_accuracy']
        explained_variance_ratios = df_result['explained_variance_ratio']
    else:
        n_components_range = range(1, 21)  # 假设测试 1 到 20 的 n_components
        mean_accuracies = []
        explained_variance_ratios = []

        for n_components in n_components_range:
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
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]
                scaler, pls = train_plsda(X_tr, y_tr, n_components=n_components)
                X_val_scaled = scaler.transform(X_val)
                pred = pls.predict(X_val_scaled)
                pred_label = np.round(pred).astype(int).flatten()
                acc = accuracy_score(y_val, pred_label)
                accs.append(acc)
            mean_acc = np.mean(accs)
            mean_accuracies.append(mean_acc)
            print(f"n_components: {n_components}, 5折交叉验证平均准确率: {mean_acc:.4f}")

            # 修复方差贡献率计算
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            pls = PLSRegression(n_components=n_components)
            pls.fit(X_scaled, y_train)
            total_variance = np.sum(np.var(X_scaled, axis=0))
            explained_variance = np.sum(np.var(pls.x_scores_, axis=0))
            explained_variance_ratios.append(explained_variance / total_variance)

        # 保存到Excel
        df_save = pd.DataFrame({
            'n_components': list(n_components_range),
            'mean_accuracy': mean_accuracies,
            'explained_variance_ratio': explained_variance_ratios
        })
        df_save.to_excel(result_file, index=False)

    # 确保 mean_accuracies 和 explained_variance_ratios 是一维列表
    mean_accuracies = np.array(mean_accuracies)
    explained_variance_ratios = np.array(explained_variance_ratios)

    # 可视化 n_components 的测试结果
    plt.figure(figsize=(12, 8))
    ax1 = plt.gca()
    color1 = 'tab:blue'
    color2 = 'tab:orange'
    # 左侧y轴：平均准确率
    l1, = ax1.plot(n_components_range, mean_accuracies, marker='o', label='平均准确率', color=color1, linewidth=2)
    ax1.set_xlabel('n_components', fontsize=14)
    ax1.set_ylabel('平均准确率', color=color1, fontsize=14)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)
    # 在n_components=13处画竖线
    ax1.axvline(x=13, color='red', linestyle='--', linewidth=2, label='最优 n_components=13')
    ax1.text(13+0.2, max(mean_accuracies)*0.95, '最优 n_components=13', color='red', fontsize=13, va='top', ha='left', rotation=90, backgroundcolor='white')
    # 右侧y轴：方差贡献率
    ax2 = ax1.twinx()
    l2, = ax2.plot(n_components_range, explained_variance_ratios, marker='s', label='方差贡献率', color=color2, linewidth=2)
    ax2.set_ylabel('方差贡献率', color=color2, fontsize=14)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.3, 0.75)
    # 合并图例
    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=13)
    plt.title('n_components 对平均准确率和方差贡献率的影响', fontsize=16)
    plt.tight_layout()
    plt.show()

    # 找到最优 n_components
    best_n_components = n_components_range[np.argmax(mean_accuracies)]
    best_accuracy = max(mean_accuracies)
    print(f"最优 n_components: {best_n_components}, 平均准确率: {best_accuracy:.4f}")