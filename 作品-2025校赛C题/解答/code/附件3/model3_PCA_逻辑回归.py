# 主成分分析+逻辑回归（结构仿照PLS-DA）
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

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

def train_pca_lr(X_train, y_train, var_ratio=0.90, model_path='pca_lr_model.pkl', cv=5):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_train)
    pca = PCA(n_components=var_ratio)
    X_train_pca = pca.fit_transform(X_train)
    model = LogisticRegressionCV(Cs=10, cv=5, max_iter=10000)
    cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    acc_list = []
    f1_list = []
    for train_idx, test_idx in cv_split.split(X_train_pca, y_encoded):
        X_tr, X_val = X_train_pca[train_idx], X_train_pca[test_idx]
        y_tr, y_val = y_encoded[train_idx], y_encoded[test_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        acc_list.append(acc)
        f1_list.append(f1)
    mean_acc = float(np.mean(acc_list)) if len(acc_list) > 0 else 0.0
    mean_f1 = float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0
    model.fit(X_train_pca, y_encoded)
    if model_path is not None:
        joblib.dump({'model': model, 'le': le, 'pca': pca, 'cv_acc_list': acc_list, 'cv_acc_mean': mean_acc, 'cv_f1_list': f1_list, 'cv_f1_mean': mean_f1}, model_path)
    return {'model': model, 'le': le, 'pca': pca, 'cv_acc_list': acc_list, 'cv_acc_mean': mean_acc, 'cv_f1_list': f1_list, 'cv_f1_mean': mean_f1, 'y_encoded': y_encoded, 'X_train_pca': X_train_pca}

def predict_pca_lr(X_test, model_path='pca_lr_model.pkl'):
    data = joblib.load(model_path)
    model = data['model']
    le = data['le']
    pca = data['pca']
    X_test_pca = pca.transform(X_test)
    y_test_pred = model.predict(X_test_pca)
    y_test_labels = le.inverse_transform(y_test_pred)
    return y_test_labels

def plot_explained_variance(pca, save_path='PCA_累计解释方差曲线.png'):
    plt.figure(figsize=(8, 5))
    sns.set(style="ticks", font_scale=1.3)
    plt.plot(np.arange(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), marker='o', color='#1f77b4', linewidth=2.5, label='PCA Cumulative Explained Variance')
    plt.xlabel('Number of Components', fontweight='bold')
    plt.ylabel('Cumulative Explained Variance', fontweight='bold')
    plt.title('PCA Cumulative Explained Variance', fontweight='bold')
    plt.ylim(0, 1.05)
    plt.legend(frameon=True)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_feature_importance(model, pca, save_path='PCA_LR_特征重要性.png'):
    if hasattr(model, 'coef_'):
        # 取绝对值和
        feature_importance = np.abs(model.coef_).sum(axis=0)
        # 投影回原始特征空间
        orig_importance = np.abs(np.dot(feature_importance, np.abs(pca.components_)))
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(orig_importance)), orig_importance)
        plt.title('Feature Importance (PCA+LR)')
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

def main_train():
    train_df = pd.read_excel('总表.xlsx')
    X_train = preprocess_data(train_df)
    y_train = train_df.iloc[:, 0].values
    result = train_pca_lr(X_train, y_train)
    model = result['model']
    le = result['le']
    pca = result['pca']
    acc_list = result['cv_acc_list']
    mean_acc = result['cv_acc_mean']
    f1_list = result['cv_f1_list']
    mean_f1 = result['cv_f1_mean']
    y_encoded = result['y_encoded']
    X_train_pca = result['X_train_pca']
    print(f"PCA累计解释方差: {np.sum(pca.explained_variance_ratio_):.2%}")
    print(f"PCA主成分数: {pca.n_components_}")
    print(f"交叉验证准确率列表: {acc_list}")
    print(f"交叉验证平均准确率: {mean_acc:.4f}")
    print(f"交叉验证F1列表: {f1_list}")
    print(f"交叉验证平均F1: {mean_f1:.4f}")
    plot_explained_variance(pca)
    plot_feature_importance(model, pca)
    train_pred = model.predict(X_train_pca)
    plot_confusion(y_encoded, train_pred, le.classes_)
    train_acc = accuracy_score(y_encoded, train_pred)
    print(f"训练集准确率: {train_acc:.4f}")

def main_test():
    test_df = pd.read_excel('测试集数据.xlsx')
    X_test = preprocess_data(test_df)
    y_test_labels = predict_pca_lr(X_test)
    result_df = pd.DataFrame({
        '样本编号': test_df.iloc[:, 0],
        '预测类别': y_test_labels
    })
    result_df.to_excel('pca结果/问题3_PCA逻辑回归结果.xlsx', index=False)
    print('测试集预测结果已保存。')

if __name__ == '__main__':
    if not os.path.exists('pca_lr_model.pkl'):
        main_train()
    main_test()