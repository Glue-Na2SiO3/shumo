import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import time
import matplotlib.pyplot as plt

def train_lightgbm(X_train, y_train, X_test=None, model_path='lgbm_model.pkl', cv=5):
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    var_thres = VarianceThreshold(threshold=0)
    X_train_var = var_thres.fit_transform(X_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_var, y_train_encoded)
    selector = SelectFromModel(rf, prefit=True, max_features=200)
    X_train_reduced = selector.transform(X_train_var)
    params = {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'random_state': 42,
        'n_jobs': -1
    }
    lgb = LGBMClassifier(**params, verbose=-1)
    cv_split = StratifiedKFold(n_splits=cv, shuffle=True)
    acc_list = []
    f1_list = []
    for train_idx, test_idx in cv_split.split(X_train_reduced, y_train_encoded):
        X_tr, X_val = X_train_reduced[train_idx], X_train_reduced[test_idx]
        y_tr, y_val = y_train_encoded[train_idx], y_train_encoded[test_idx]
        lgb.fit(X_tr, y_tr)
        y_pred = lgb.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        acc_list.append(acc)
        f1_list.append(f1)
    mean_acc = float(np.mean(acc_list)) if len(acc_list) > 0 else 0.0
    mean_f1 = float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0
    lgb.fit(X_train_reduced, y_train_encoded)
    if model_path is not None:
        import joblib
        joblib.dump({'model': lgb, 'le': le, 'var_thres': var_thres, 'selector': selector, 'cv_acc_list': acc_list, 'cv_acc_mean': mean_acc, 'cv_f1_list': f1_list, 'cv_f1_mean': mean_f1}, model_path)
    return {'model': lgb, 'le': le, 'var_thres': var_thres, 'selector': selector, 'cv_acc_list': acc_list, 'cv_acc_mean': mean_acc, 'cv_f1_list': f1_list, 'cv_f1_mean': mean_f1}

def predict_lightgbm(X_test, model_path='lgbm_model.pkl'):
    import joblib
    data = joblib.load(model_path)
    model = data['model']
    le = data['le']
    var_thres = data['var_thres']
    selector = data['selector']
    X_test_reduced = selector.transform(var_thres.transform(X_test))
    y_test_pred = model.predict(X_test_reduced)
    y_test_labels = le.inverse_transform(y_test_pred)
    return y_test_labels

def main_train():
    train_df = pd.read_excel('总表.xlsx')
    X_train = train_df.iloc[:, 1:].values
    y_train = train_df.iloc[:, 0].values
    result = train_lightgbm(X_train, y_train)
    print(f"5折交叉验证准确率: {result['cv_acc_list']}")
    print(f"平均准确率: {result['cv_acc_mean']:.4f}")
    print(f"5折交叉验证F1: {result['cv_f1_list']}")
    print(f"平均F1: {result['cv_f1_mean']:.4f}")
    train_pred = result['model'].predict(result['selector'].transform(result['var_thres'].transform(X_train)))
    from sklearn.metrics import ConfusionMatrixDisplay
    ConfusionMatrixDisplay.from_estimator(result['model'], result['selector'].transform(result['var_thres'].transform(X_train)), result['le'].transform(y_train))
    plt.title('Confusion Matrix')
    plt.show()
    train_acc = accuracy_score(result['le'].transform(y_train), train_pred)
    print(f"训练集准确率: {train_acc:.4f}")
    importances = result['model'].feature_importances_
    plt.figure()
    plt.bar(range(len(importances)), importances)
    plt.title('Feature Importances')
    plt.show()

def main_test():
    test_df = pd.read_excel('测试集数据.xlsx')
    X_test = test_df.iloc[:, 1:].values
    y_test_labels = predict_lightgbm(X_test)
    result_df = pd.DataFrame({
        '样本编号': test_df.iloc[:, 0],
        '预测类别': y_test_labels
    })
    result_df.to_excel('问题2_LightGBM结果.xlsx', index=False)
    print('测试集预测结果已保存。')

if __name__ == '__main__':
    import os
    if not os.path.exists('lgbm_model.pkl'):
        main_train()
    main_test()