import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def preprocess_features(X):
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d
    # 先clip到0-100
    X = np.clip(X, 0, 100)
    # savgol滤波
    X = savgol_filter(X, window_length=3, polyorder=2, axis=1, mode='nearest')

    return X

def train_svm(X_train, y_train, model_path='svm_model.pkl', cv=5):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    selector = VarianceThreshold(threshold=0)
    X_train_selected = selector.fit_transform(X_train)
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 0.01, 0.1],
        'kernel': ['rbf']
    }
    svm = SVC(probability=True)
    grid_search = GridSearchCV(svm, param_grid, cv=cv, n_jobs=-1, return_train_score=True)
    grid_search.fit(X_train_selected, y_encoded)
    best_model = grid_search.best_estimator_
    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    acc_list = []
    f1_list = []
    for train_idx, test_idx in cv_split.split(X_train_selected, y_encoded):
        X_tr, X_val = X_train_selected[train_idx], X_train_selected[test_idx]
        y_tr, y_val = y_encoded[train_idx], y_encoded[test_idx]
        best_model.fit(X_tr, y_tr)
        y_pred = best_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        acc_list.append(acc)
        f1_list.append(f1)
    mean_acc = float(np.mean(acc_list)) if len(acc_list) > 0 else 0.0
    mean_f1 = float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0
    if model_path is not None:
        joblib.dump({'model': best_model, 'le': le, 'selector': selector, 'cv_acc_list': acc_list, 'cv_acc_mean': mean_acc, 'cv_f1_list': f1_list, 'cv_f1_mean': mean_f1, 'cv_results': grid_search.cv_results_}, model_path)
    return {'model': best_model, 'le': le, 'selector': selector, 'cv_acc_list': acc_list, 'cv_acc_mean': mean_acc, 'cv_f1_list': f1_list, 'cv_f1_mean': mean_f1, 'y_encoded': y_encoded, 'X_train_selected': X_train_selected, 'cv_results': grid_search.cv_results_}

def predict_svm(X_test, model_path='svm_model.pkl'):
    data = joblib.load(model_path)
    model = data['model']
    le = data['le']
    selector = data['selector']
    X_test_selected = selector.transform(X_test)
    y_test_pred = model.predict(X_test_selected)
    y_test_labels = le.inverse_transform(y_test_pred)
    return y_test_labels

def plot_feature_importance(model, selector, save_path='SVM_特征重要性.png'):
    if hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_).sum(axis=0)
        orig_idx = np.where(selector.get_support())[0]
        full_importance = np.zeros(selector.transform(np.zeros((1, selector.n_features_in_))).shape[1])
        full_importance[:len(feature_importance)] = feature_importance
        plt.figure(figsize=(10, 6))
        plt.bar(orig_idx, full_importance)
        plt.title('Feature Importance (SVM Coefficients)')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

def plot_confusion(y_true, y_pred, labels, save_path=None):
    conf_matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
    disp.plot()
    plt.title('Confusion Matrix on Training Set')
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_grid_search_heatmap(cv_results, param_grid, save_path='SVM超参数热力图.png'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # 只考虑kernel=rbf的情况
    results = cv_results
    scores = results['mean_test_score']
    params = results['params']
    C_list = param_grid['C']
    gamma_list = param_grid['gamma']
    score_matrix = np.zeros((len(C_list), len(gamma_list)))
    for i, C in enumerate(C_list):
        for j, gamma in enumerate(gamma_list):
            for idx, p in enumerate(params):
                if p['C'] == C and p['gamma'] == gamma and p['kernel'] == 'rbf':
                    score_matrix[i, j] = scores[idx]
    plt.figure(figsize=(8, 6))
    sns.heatmap(score_matrix, annot=True, fmt='.3f', xticklabels=gamma_list, yticklabels=C_list, cmap='YlGnBu')
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.title('SVM超参数网格搜索准确率热力图')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main_train():
    train_df = pd.read_excel('总表.xlsx')
    X_train = train_df.iloc[:, 1:].values
    X_train = preprocess_features(X_train)
    y_train = train_df.iloc[:, 0].values
    result = train_svm(X_train, y_train)
    best_model = result['model']
    le = result['le']
    selector = result['selector']
    acc_list = result['cv_acc_list']
    mean_acc = result['cv_acc_mean']
    f1_list = result['cv_f1_list']
    mean_f1 = result['cv_f1_mean']
    y_encoded = result['y_encoded']
    X_train_selected = result['X_train_selected']
    cv_results = result['cv_results']
    print(f"交叉验证准确率列表: {acc_list}")
    print(f"交叉验证平均准确率: {mean_acc:.4f}")
    print(f"交叉验证F1列表: {f1_list}")
    print(f"交叉验证平均F1: {mean_f1:.4f}")
    train_pred = best_model.predict(X_train_selected)
    plot_confusion(y_encoded, train_pred, le.classes_)
    train_acc = accuracy_score(y_encoded, train_pred)
    print(f"训练集准确率: {train_acc:.4f}")
    plot_feature_importance(best_model, selector)
    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1], 'kernel': ['rbf']}
    plot_grid_search_heatmap(cv_results, param_grid)

def main_test():
    test_df = pd.read_excel('测试集数据.xlsx')
    X_test = test_df.iloc[:, 1:].values
    X_test = preprocess_features(X_test)
    y_test_labels = predict_svm(X_test)
    result_df = pd.DataFrame({
        '样本编号': test_df.iloc[:, 0],
        '预测类别': y_test_labels
    })
    # result_df.to_excel('SVM结果/问题3_SVM可视化结果.xlsx', index=False)
    print('测试集预测结果已保存。')

if __name__ == '__main__':
    main_train()
    # if not os.path.exists('svm_model.pkl'):
    #     main_train()
    # main_test()