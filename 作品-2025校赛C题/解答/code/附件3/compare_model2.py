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
import os
logging.getLogger('lightgbm').setLevel(logging.ERROR)  # 设置 LightGBM 日志级别为 ERROR
warnings.filterwarnings('ignore')

# 导入各模型的train和predict函数
from model3_LightGBM import train_lightgbm, predict_lightgbm
from model3_SVM import train_svm, predict_svm
from model3_随机森林_RFE import train_rf_rfe, predict_rf_rfe
from model3_PCA_逻辑回归 import train_pca_lr, predict_pca_lr
from model3_PLS_DA import train_plsda, predict_plsda
from model3_XGBoost增量学习 import train_xgboost, predict_xgboost
from model3_Stacking集成 import train_stacking, predict_stacking
from model3_VAE降维 import train_vae, predict_vae

# === 预处理函数，与model3_SVM保持一致 ===
def preprocess_features(X):
    from scipy.signal import savgol_filter
    from scipy.ndimage import gaussian_filter1d
    X = np.clip(X, 0, 100)
    X = savgol_filter(X, window_length=3, polyorder=2, axis=1, mode='nearest')
    return X

# 数据路径配置
train_file = '总表.xlsx'
# 只加载训练集数据
train_df = pd.read_excel(train_file)
X = train_df.iloc[:, 1:].values
X = preprocess_features(X)  # 加入预处理
y = train_df.iloc[:, 0].values

results = []

# 统一模型对比流程
model_funcs = [
    ('LightGBM', train_lightgbm, predict_lightgbm, 'lgbm_model.pkl', '问题2_LightGBM对比结果.xlsx'),
    ('SVM', train_svm, predict_svm, 'svm_model.pkl', '问题2_SVM对比结果.xlsx'),
    ('RF-RFE', train_rf_rfe, predict_rf_rfe, 'rf_rfe_model.pkl', '问题2_RF-RFE对比结果.xlsx'),
    ('PCA+LR', train_pca_lr, predict_pca_lr, 'pca_lr_model.pkl', '问题2_PCA+LR对比结果.xlsx'),
    ('PLS-DA', train_plsda, predict_plsda, 'plsda_model.pkl', '问题2_PLS-DA对比结果.xlsx'),
    ('XGBoost', train_xgboost, predict_xgboost, 'xgb_model.pkl', '问题2_XGBoost对比结果.xlsx'),
    ('Stacking', train_stacking, predict_stacking, 'stacking_model.pkl', '问题2_Stacking对比结果.xlsx'),
    ('VAE', train_vae, predict_vae, 'vae_model.pkl', '问题2_VAE对比结果.xlsx'),
]

for name, train_func, predict_func, model_path, out_xlsx in model_funcs:
    print(f'正在训练和五折交叉验证: {name}')
    # 只做五折交叉验证，不保存模型，不做测试集预测
    model_info = train_func(X, y, model_path=None, cv=5)
    # 兼容不同模型返回格式，要求每个模型返回dict，包含如下key
    # 'cv_acc_list', 'cv_acc_mean', 'cv_f1_list', 'cv_f1_mean'
    if isinstance(model_info, dict):
        acc_list = model_info.get('cv_acc_list', [])
        mean_acc = model_info.get('cv_acc_mean', 0.0)
        f1_list = model_info.get('cv_f1_list', [])
        mean_f1 = model_info.get('cv_f1_mean', 0.0)
    else:
        acc_list = model_info[3] if len(model_info) > 3 else []
        mean_acc = model_info[4] if len(model_info) > 4 else 0.0
        f1_list = model_info[5] if len(model_info) > 5 else []
        mean_f1 = model_info[6] if len(model_info) > 6 else 0.0
    # 保证准确率和f1有值
    if len(acc_list) == 0 or np.isnan(mean_acc):
        acc_list = [0.0]*5
        mean_acc = 0.0
    if len(f1_list) == 0 or np.isnan(mean_f1):
        f1_list = [0.0]*5
        mean_f1 = 0.0
    print(f'{name} 5折准确率: {acc_list}, 平均准确率: {mean_acc}')
    print(f'{name} 5折F1: {f1_list}, 平均F1: {mean_f1}')
    results.append({'模型': name, '5折准确率': acc_list, '平均准确率': mean_acc, '5折F1': f1_list, '平均F1': mean_f1})

result_df = pd.DataFrame(results)
result_df.to_excel('模型对比结果.xlsx', index=False)
print(result_df)