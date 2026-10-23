import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFECV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression  # 修复 model_stacking 的导入问题
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras import layers
from scikeras.wrappers import KerasClassifier  # 用于包装 Keras 模型
import warnings
import gc
import lightgbm as lgb
import logging
logging.getLogger('lightgbm').setLevel(logging.ERROR)  # 设置 LightGBM 日志级别为 ERROR
warnings.filterwarnings('ignore')

# 数据路径配置
train_file = '附件2处理/总表.xlsx'
test_file = '附件2处理/问题2测试集.xlsx'

# 数据加载
train_df = pd.read_excel(train_file)
test_df = pd.read_excel(test_file)

# 全局数据初始化
X_raw = train_df.iloc[:, 1:].values
y_raw = train_df.iloc[:, 0].values
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)

# 测试集处理
X_test = test_df.iloc[:, 1:].values

# 结果存储
results = []

# ================== 优化后的模型定义 ==================


def model_lightgbm():
    """优化后的LightGBM模型，修复警告问题"""
    start = time.time()
    # 特征选择
    var_thres = VarianceThreshold(threshold=0.1)
    X_var = var_thres.fit_transform(X_raw)
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), prefit=False, max_features=100)
    X_red = selector.fit_transform(X_var, y_encoded)
    selected_indices = var_thres.get_support(indices=True)[selector.get_support()]

    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        min_gain_to_split=0.01,  # 增加最小分裂增益
        min_data_in_leaf=5,  # 增加叶子节点的最小数据量
        random_state=42,
        n_jobs=-1,
        verbose=-1  # 屏蔽日志
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_red, y_encoded, cv=cv)

    training_time = time.time() - start
    return {
        'model_name': 'LightGBM',
        'train_acc': accuracy_score(y_encoded, y_pred),
        'val_acc': None,
        'key_wavelengths': selected_indices,
        'training_time': training_time,
        'model': model
    }

def model_pca_lr():
    """优化后的PCA+逻辑回归"""
    start = time.time()
    pca = PCA(n_components=0.90)
    X_pca = pca.fit_transform(X_raw)

    model = LogisticRegressionCV(cv=3, max_iter=10000, multi_class='multinomial')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_pca, y_encoded, cv=cv)

    training_time = time.time() - start
    return {
        'model_name': 'PCA+LR',
        'train_acc': accuracy_score(y_encoded, y_pred),
        'val_acc': None,
        'key_wavelengths': None,
        'training_time': training_time,
        'model': model
    }

def model_pls_da():
    """优化后的PLS-DA模型"""
    start = time.time()
    X_scaled = X_raw

    best_acc = 0
    best_n = 2
    for n in range(2, 11):
        pls = PLSRegression(n_components=n)
        y_pred_continuous = cross_val_predict(pls, X_scaled, y_encoded, cv=5)
        y_pred = np.clip(np.round(y_pred_continuous), 0, len(np.unique(y_encoded))-1).astype(int)
        acc = accuracy_score(y_encoded, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_n = n

    model = PLSRegression(n_components=best_n)
    model.fit(X_scaled, y_encoded)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_continuous = cross_val_predict(model, X_scaled, y_encoded, cv=cv)
    y_pred = np.clip(np.round(y_pred_continuous), 0, len(np.unique(y_encoded))-1).astype(int)

    training_time = time.time() - start
    return {
        'model_name': 'PLS-DA',
        'train_acc': accuracy_score(y_encoded, y_pred),
        'val_acc': None,
        'training_time': training_time,
        'key_wavelengths': None,
        'model': model
    }

def model_stacking():
    """优化后的堆叠模型"""
    start = time.time()
    base_models = [
        ('svm', SVC(kernel='rbf', probability=True, C=1)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5)),
        ('xgb', XGBClassifier(learning_rate=0.05, n_estimators=100))
    ]

    model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=10000),
        cv=5,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_raw, y_encoded, cv=cv)

    training_time = time.time() - start
    return {
        'model_name': 'Stacking',
        'train_acc': accuracy_score(y_encoded, y_pred),
        'val_acc': None,
        'key_wavelengths': None,
        'training_time': training_time,
        'model': model
    }

def model_svm():
    """优化后的SVM模型"""
    start = time.time()
    X_scaled = X_raw
    
    model = SVC(kernel='rbf', C=5, gamma='scale', random_state=42, probability=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_scaled, y_encoded, cv=cv)
    
    training_time = time.time() - start
    return {
        'model_name': 'SVM',
        'train_acc': accuracy_score(y_encoded, y_pred),
        'val_acc': None,
        'key_wavelengths': None,
        'training_time': training_time,
        'model': model
    }

def model_rf_rfe():
    """优化后的RF-RFE模型"""
    start = time.time()
    selector = RFECV(
        estimator=RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        step=25,
        cv=5,
        min_features_to_select=50,
        n_jobs=-1
    )
    X_red = selector.fit_transform(X_raw, y_encoded)

    model = RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_red, y_encoded, cv=cv)

    training_time = time.time() - start
    return {
        'model_name': 'RF-RFE',
        'train_acc': accuracy_score(y_encoded, y_pred),
        'val_acc': None,
        'key_wavelengths': selector.get_support(indices=True),
        'training_time': training_time,
        'model': model
    }

def model_vae():
    """优化后的VAE模型，修复输出维度问题"""
    start = time.time()
    # 构建VAE
    latent_dim = 64
    encoder = tf.keras.Sequential([
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(latent_dim)
    ])
    X_encoded = encoder.predict(X_raw.astype(np.float32))
    if X_encoded.ndim == 1:
        X_encoded = np.expand_dims(X_encoded, axis=1)  # 确保输出为二维

    # 分类模型
    classifier = LGBMClassifier()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(classifier, X_encoded, y_encoded, cv=cv)

    training_time = time.time() - start
    return {
        'model_name': 'VAE',
        'train_acc': accuracy_score(y_encoded, y_pred),
        'val_acc': None,
        'key_wavelengths': None,
        'training_time': training_time,
        'model': classifier
    }

def model_xgboost():
    """优化后的XGBoost模型"""
    start = time.time()
    model = XGBClassifier(tree_method='hist', enable_categorical=True, n_estimators=200, max_depth=6, learning_rate=0.1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_raw, y_encoded, cv=cv)

    training_time = time.time() - start
    return {
        'model_name': 'XGBoost',
        'train_acc': accuracy_score(y_encoded, y_pred),
        'val_acc': None,
        'key_wavelengths': None,
        'training_time': training_time,
        'model': model
    }

# 添加资源释放和独立运行逻辑
# 定义一个通用的运行函数
def run_model(model_func, *args, **kwargs):
    try:
        print(f"正在运行模型: {model_func.__name__}")
        result = model_func(*args, **kwargs)
        print(f"模型 {model_func.__name__} 运行完成")
        return result
    except Exception as e:
        print(f"模型 {model_func.__name__} 运行失败: {e}")
        return {
            'model_name': model_func.__name__,
            'train_acc': None,
            'val_acc': None,
            'training_time': None,
            'key_wavelengths': None
        }
    finally:
        gc.collect()
        tf.keras.backend.clear_session()

# 修改主逻辑，确保结果存储完整性
if __name__ == "__main__":
    models = [
        model_lightgbm,
        model_svm,
        model_rf_rfe,
        model_pca_lr,
        model_pls_da,
        model_xgboost,
        model_stacking,
        model_vae
    ]

    results = []
    for model_func in models:
        result = run_model(model_func)
        results.append(result)

    # 检查结果是否为空
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df.to_excel('模型运行结果.xlsx', index=False)

        # 可视化通用指标
        import matplotlib.pyplot as plt

        metrics = ['train_acc']  # 仅保留训练准确率
        for metric in metrics:
            if metric in result_df.columns and result_df[metric].notnull().any():
                plt.figure(figsize=(10, 6))
                plt.bar(result_df['model_name'], result_df[metric].fillna(0), color='skyblue')
                plt.title(f'Model Comparison by {metric}')
                plt.xlabel('Model')
                plt.ylabel(metric)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'{metric}_comparison.png')
                plt.show()

        # 模型选择建议
        if 'train_acc' in result_df.columns and result_df['train_acc'].notnull().any():
            best_model = result_df.loc[result_df['train_acc'].idxmax()]
            print(f"推荐模型: {best_model['model_name']}，训练准确率最高: {best_model['train_acc']:.4f}")
        else:
            print("无法推荐模型，因为没有有效的训练准确率。")
    else:
        print("没有成功运行的模型，无法生成结果。")