# 随机森林递归特征消除
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
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

def train_rf_rfe(X_train, y_train, model_path='rf_rfe_model.pkl', cv=5):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    estimator = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    selector = RFECV(estimator, step=25, cv=cv, min_features_to_select=50, n_jobs=-1)
    X_train_selected = selector.fit_transform(X_train, y_encoded)
    model = RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    acc_list = []
    f1_list = []
    for train_idx, test_idx in skf.split(X_train_selected, y_encoded):
        X_tr, X_val = X_train_selected[train_idx], X_train_selected[test_idx]
        y_tr, y_val = y_encoded[train_idx], y_encoded[test_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        acc_list.append(acc)
        f1_list.append(f1)
    mean_acc = float(np.mean(acc_list)) if len(acc_list) > 0 else 0.0
    mean_f1 = float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0
    model.fit(X_train_selected, y_encoded)
    import joblib
    if model_path is not None:
        joblib.dump({'model': model, 'le': le, 'selector': selector, 'cv_acc_list': acc_list, 'cv_acc_mean': mean_acc, 'cv_f1_list': f1_list, 'cv_f1_mean': mean_f1}, model_path)
    return {'model': model, 'le': le, 'selector': selector, 'cv_acc_list': acc_list, 'cv_acc_mean': mean_acc, 'cv_f1_list': f1_list, 'cv_f1_mean': mean_f1, 'y_encoded': y_encoded, 'X_train_selected': X_train_selected}

def predict_rf_rfe(X_test, model_path='rf_rfe_model.pkl'):
    import joblib
    data = joblib.load(model_path)
    model = data['model']
    le = data['le']
    selector = data['selector']
    X_test_selected = selector.transform(X_test)
    y_test_pred = model.predict(X_test_selected)
    y_test_labels = le.inverse_transform(y_test_pred)
    return y_test_labels

def main_train():
    train_df = pd.read_excel('总表.xlsx')
    X_train = preprocess_data(train_df)
    y_train = train_df.iloc[:, 0].values
    result = train_rf_rfe(X_train, y_train)
    model = result['model']
    le = result['le']
    selector = result['selector']
    acc_list = result['cv_acc_list']
    mean_acc = result['cv_acc_mean']
    f1_list = result['cv_f1_list']
    mean_f1 = result['cv_f1_mean']
    y_encoded = result['y_encoded']
    X_train_selected = result['X_train_selected']
    print(f"5折交叉验证准确率: {acc_list}")
    print(f"平均交叉验证准确率: {mean_acc:.4f}")
    print(f"5折交叉验证F1: {f1_list}")
    print(f"平均交叉验证F1: {mean_f1:.4f}")
    train_pred = model.predict(X_train_selected)
    ConfusionMatrixDisplay.from_estimator(model, X_train_selected, y_encoded)
    plt.title('Confusion Matrix')
    plt.show()
    importances = model.feature_importances_
    plt.figure()
    plt.bar(range(len(importances)), importances)
    plt.title('Feature Importances')
    plt.show()
    train_acc = accuracy_score(y_encoded, train_pred)
    print(f"训练集准确率: {train_acc:.4f}")

def main_test():
    test_df = pd.read_excel('测试集数据.xlsx')
    X_test = preprocess_data(test_df)
    y_test_labels = predict_rf_rfe(X_test)
    result_df = pd.DataFrame({
        '样本编号': test_df.iloc[:, 0],
        '预测类别': y_test_labels
    })
    result_df.to_excel('问题2_RFE随机森林结果.xlsx', index=False)
    print('测试集预测结果已保存。')

if __name__ == '__main__':
    import os
    if not os.path.exists('rf_rfe_model.pkl'):
        main_train()
    main_test()