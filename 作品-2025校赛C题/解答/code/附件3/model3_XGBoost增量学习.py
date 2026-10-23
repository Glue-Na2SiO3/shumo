# XGBoost流式处理
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
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

def train_xgboost(X_train, y_train, model_path='xgb_model.pkl', cv=5):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    param_dist = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.05]
    }
    model = XGBClassifier(tree_method='hist', enable_categorical=True)
    search = RandomizedSearchCV(model, param_dist, n_iter=5, cv=3, n_jobs=-1)
    search.fit(X_train, y_encoded)
    best_model = search.best_estimator_
    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    acc_list = []
    f1_list = []
    for train_idx, test_idx in cv_split.split(X_train, y_encoded):
        X_tr, X_val = X_train[train_idx], X_train[test_idx]
        y_tr, y_val = y_encoded[train_idx], y_encoded[test_idx]
        best_model.fit(X_tr, y_tr)
        y_pred = best_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        acc_list.append(acc)
        f1_list.append(f1)
    mean_acc = float(np.mean(acc_list)) if len(acc_list) > 0 else 0.0
    mean_f1 = float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0
    import joblib
    if model_path is not None:
        joblib.dump({'model': best_model, 'le': le, 'cv_acc_list': acc_list, 'cv_acc_mean': mean_acc, 'cv_f1_list': f1_list, 'cv_f1_mean': mean_f1}, model_path)
    return {'model': best_model, 'le': le, 'cv_acc_list': acc_list, 'cv_acc_mean': mean_acc, 'cv_f1_list': f1_list, 'cv_f1_mean': mean_f1, 'y_encoded': y_encoded, 'X_train': X_train}

def predict_xgboost(X_test, model_path='xgb_model.pkl'):
    import joblib
    data = joblib.load(model_path)
    model = data['model']
    le = data['le']
    y_test_pred = model.predict(X_test)
    y_test_labels = le.inverse_transform(y_test_pred)
    return y_test_labels

def main_train():
    train_df = pd.read_excel('总表.xlsx')
    X_train = preprocess_data(train_df)
    y_train = train_df.iloc[:, 0].values
    result = train_xgboost(X_train, y_train)
    model = result['model']
    le = result['le']
    acc_list = result['cv_acc_list']
    mean_acc = result['cv_acc_mean']
    f1_list = result['cv_f1_list']
    mean_f1 = result['cv_f1_mean']
    y_encoded = result['y_encoded']
    X_train_used = result['X_train']
    print(f"5折交叉验证准确率: {acc_list}")
    print(f"5折交叉验证平均准确率: {mean_acc:.4f}")
    print(f"5折交叉验证F1分数: {f1_list}")
    print(f"5折交叉验证平均F1分数: {mean_f1:.4f}")
    train_pred = model.predict(X_train_used)
    ConfusionMatrixDisplay.from_estimator(model, X_train_used, y_encoded)
    plt.title('Confusion Matrix')
    plt.show()
    train_acc = accuracy_score(y_encoded, train_pred)
    print(f"训练集准确率: {train_acc:.4f}")
    from xgboost import plot_importance
    plot_importance(model, max_num_features=20)
    plt.show()

def main_test():
    test_df = pd.read_excel('测试集数据.xlsx')
    X_test = preprocess_data(test_df)
    y_test_labels = predict_xgboost(X_test)
    result_df = pd.DataFrame({
        '样本编号': test_df.iloc[:, 0],
        '预测类别': y_test_labels
    })
    result_df.to_excel('问题2_XGBoost结果.xlsx', index=False)
    print('测试集预测结果已保存。')

if __name__ == '__main__':
    import os
    if not os.path.exists('xgb_model.pkl'):
        main_train()
    main_test()