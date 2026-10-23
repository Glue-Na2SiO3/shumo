import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from scipy.signal import savgol_filter
import joblib

def read_data(file_path):
    df = pd.read_excel(file_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:]
    X = X.apply(pd.to_numeric, errors='coerce')
    # 先clip到[0, 100]区间
    X = X.clip(lower=0, upper=100)
    # 对每一列用线性插值填补clip后的缺失值
    X = X.interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill')
    X = X.values.astype(float)
    # SG滤波
    X = np.array([savgol_filter(row, window_length=3, polyorder=2) for row in X])
    return X, y

def train_plsda(X, y, n_components=13, model_path='plsda_model.pkl', cv=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_scaled, y)
    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    accs = []
    f1s = []
    for train_idx, val_idx in cv_split.split(X_scaled, y):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        pls_cv = PLSRegression(n_components=n_components)
        pls_cv.fit(X_tr, y_tr)
        pred = pls_cv.predict(X_val)
        pred_label = np.round(pred).astype(int).flatten()
        acc = accuracy_score(y_val, pred_label)
        f1 = f1_score(y_val, pred_label, average='weighted')
        accs.append(acc)
        f1s.append(f1)
    mean_acc = np.mean(accs)
    mean_f1 = np.mean(f1s)
    if model_path is not None:
        joblib.dump({'scaler': scaler, 'pls': pls, 'cv_acc_list': accs, 'cv_acc_mean': mean_acc, 'cv_f1_list': f1s, 'cv_f1_mean': mean_f1}, model_path)
    return {'scaler': scaler, 'pls': pls, 'cv_acc_list': accs, 'cv_acc_mean': mean_acc, 'cv_f1_list': f1s, 'cv_f1_mean': mean_f1}

def predict_plsda(X_test, model_path='plsda_model.pkl'):
    import joblib
    data = joblib.load(model_path)
    scaler = data['scaler']
    pls = data['pls']
    X_test_scaled = scaler.transform(X_test)
    pred = pls.predict(X_test_scaled)
    pred_label = np.round(pred).astype(int).flatten()
    return pred_label

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
    # 固定参数
    window_length = 3
    polyorder = 2
    n_components = 13

    # 读取数据
    X, y = read_data("总表.xlsx")

    # 5折交叉验证
    model = train_plsda(X, y, n_components=n_components, cv=5)
    accs = model['cv_acc_list']
    mean_acc = model['cv_acc_mean']
    mean_f1 = model['cv_f1_mean']
    for i, acc in enumerate(accs):
        print(f"第{i+1}折准确率: {acc:.4f}")
    print(f"5折交叉验证平均准确率: {mean_acc:.4f}")
    print(f"5折交叉验证平均F1: {mean_f1:.4f}")

    # 调用预测并保存结果
    predict_and_save_plsda((model['scaler'], model['pls']), "测试集数据.xlsx", "第三题分类结果.xlsx")