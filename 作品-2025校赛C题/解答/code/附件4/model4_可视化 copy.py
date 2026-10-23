# -*- coding: utf-8 -*-
"""
可视化分析模块
1. 数据预处理效果可视化
2. 特征工程分析可视化
3. 模型性能评估可视化
4. 结果展示可视化
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import pywt
import joblib

# 设置全局样式
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 可视化数据保存路径
VIZ_DATA_PATH = "visualization_data.pkl"

def save_viz_data(data_dict):
    """保存可视化数据到文件"""
    joblib.dump(data_dict, VIZ_DATA_PATH)
    print(f"[VIZ] 可视化数据已保存到 {VIZ_DATA_PATH}")

def load_viz_data():
    """从文件加载可视化数据"""
    if os.path.exists(VIZ_DATA_PATH):
        data = joblib.load(VIZ_DATA_PATH)
        print(f"[VIZ] 从 {VIZ_DATA_PATH} 加载可视化数据")
        return data
    return None

# ======================== 数据预处理可视化 ========================
def plot_preprocessing_effects(X_raw, X_preprocessed, wavelengths, sample_idx=0):
    """
    绘制预处理前后光谱对比
    """
    plt.figure(figsize=(12, 6))
    plt.plot(wavelengths, X_raw[sample_idx], 'b-', alpha=0.5, label='原始光谱')
    plt.plot(wavelengths, X_preprocessed[sample_idx], 'r-', label='预处理后')
    plt.title(f"样本 {sample_idx} 预处理前后光谱对比", fontsize=14)
    plt.xlabel("波长(nm)", fontsize=12)
    plt.ylabel("吸光度", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("preprocessing_effects.png", dpi=300)
    plt.show()

# ======================== 特征工程可视化 ========================
def plot_feature_engineering(features, y_train, wavelengths):
    """
    特征工程可视化
    1. 特征分布箱线图
    2. PCA降维可视化
    """
    # 特征分布箱线图
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=pd.DataFrame(features[:, :20]), palette="Set3")
    plt.title("前20个特征分布箱线图", fontsize=14)
    plt.xlabel("特征索引", fontsize=12)
    plt.ylabel("特征值", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("feature_distribution.png", dpi=300)
    plt.show()
    
    # PCA降维可视化
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=y_train, 
                         cmap='viridis', alpha=0.7, edgecolors='w')
    plt.title("特征PCA降维可视化 (前2主成分)", fontsize=14)
    plt.xlabel(f"PC1 (方差解释率: {pca.explained_variance_ratio_[0]:.2f})", fontsize=12)
    plt.ylabel(f"PC2 (方差解释率: {pca.explained_variance_ratio_[1]:.2f})", fontsize=12)
    plt.colorbar(scatter, label='类别标签')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("feature_pca.png", dpi=300)
    plt.show()

# ======================== 模型性能可视化 ========================
def plot_model_performance(y_true, y_pred, acc_scores):
    """
    模型性能可视化
    1. 混淆矩阵
    2. 交叉验证准确率
    """
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_true), 
                yticklabels=np.unique(y_true))
    plt.title("混淆矩阵", fontsize=14)
    plt.xlabel("预测标签", fontsize=12)
    plt.ylabel("真实标签", fontsize=12)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()
    
    # 交叉验证准确率
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(acc_scores)+1), acc_scores, color='skyblue', alpha=0.7)
    plt.axhline(np.mean(acc_scores), color='red', linestyle='--', 
                label=f'平均准确率: {np.mean(acc_scores):.4f}')
    plt.title("5折交叉验证准确率", fontsize=14)
    plt.xlabel("折数", fontsize=12)
    plt.ylabel("准确率", fontsize=12)
    plt.ylim(0.8, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cv_accuracy.png", dpi=300)
    plt.show()

# ======================== 特征重要性可视化 ========================
def plot_feature_importance(model, feature_names=None, top_n=20):
    """
    绘制XGBoost特征重要性
    """
    pls, xgb, _, _ = model
    importance = xgb.feature_importances_
    
    if feature_names is None:
        feature_names = [f'特征_{i}' for i in range(len(importance))]
    
    # 按重要性排序
    indices = np.argsort(importance)[-top_n:]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), importance[indices], color='teal', alpha=0.7)
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.title(f"Top {top_n} 重要特征", fontsize=14)
    plt.xlabel("特征重要性", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=300)
    plt.show()

# ======================== 核PLS潜变量可视化 ========================
def plot_latent_variables(model, X, y):
    """
    核PLS潜变量可视化
    """
    pls, _, X_ref, gamma = model
    K = rbf_kernel(X, X_ref, gamma=gamma)
    T = pls.transform(K)
    
    plt.figure(figsize=(12, 5))
    
    # 第一对潜变量
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(T[:, 0], T[:, 1], c=y, cmap='viridis', alpha=0.7)
    plt.title("第一对潜变量 (LV1 vs LV2)", fontsize=12)
    plt.xlabel("LV1", fontsize=10)
    plt.ylabel("LV2", fontsize=10)
    plt.colorbar(scatter, label='类别标签')
    plt.grid(True, alpha=0.3)
    
    # 第二对潜变量
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(T[:, 2], T[:, 3], c=y, cmap='viridis', alpha=0.7)
    plt.title("第二对潜变量 (LV3 vs LV4)", fontsize=12)
    plt.xlabel("LV3", fontsize=10)
    plt.ylabel("LV4", fontsize=10)
    plt.colorbar(scatter, label='类别标签')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle("核PLS潜变量可视化", fontsize=14)
    plt.tight_layout()
    plt.savefig("latent_variables.png", dpi=300)
    plt.show()

# ======================== 主可视化流程 ========================
def main_visualization():
    # 尝试加载已有可视化数据
    viz_data = load_viz_data()
    
    if viz_data is None:
        print("[VIZ] 未找到可视化数据文件，将重新计算...")
        
        # 这里应该从您的原始代码中获取数据
        # 示例数据生成 (实际使用时替换为您的数据)
        from sklearn.datasets import make_classification
        X_raw, y_train = make_classification(n_samples=100, n_features=50, n_classes=3, random_state=42)
        wavelengths = np.linspace(400, 1000, 50)
        
        # 预处理
        X_preprocessed = preprocess(X_raw.copy())
        
        # 特征工程
        features = extract_nonlinear_features(X_preprocessed, wavelengths)
        
        # 训练模型
        model = train_kplsda_xgb(features, y_train)
        
        # 交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        acc_scores = []
        y_preds = []
        y_trues = []
        
        for train_idx, val_idx in skf.split(features, y_train):
            X_tr, X_val = features[train_idx], features[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model_cv = train_kplsda_xgb(X_tr, y_tr)
            pls, xgb, X_ref, gamma = model_cv
            
            K_val = rbf_kernel(X_val, X_ref, gamma=gamma)
            T_val = pls.transform(K_val)
            X_val_combined = np.hstack([T_val, X_val])
            
            y_pred = xgb.predict(X_val_combined)
            acc_scores.append(accuracy_score(y_val, y_pred))
            y_preds.extend(y_pred)
            y_trues.extend(y_val)
        
        # 保存可视化数据
        viz_data = {
            'X_raw': X_raw,
            'X_preprocessed': X_preprocessed,
            'wavelengths': wavelengths,
            'features': features,
            'y_train': y_train,
            'model': model,
            'acc_scores': acc_scores,
            'y_preds': y_preds,
            'y_trues': y_trues
        }
        save_viz_data(viz_data)
    
    # 绘制各种可视化图表
    plot_preprocessing_effects(viz_data['X_raw'], viz_data['X_preprocessed'], 
                              viz_data['wavelengths'])
    
    plot_feature_engineering(viz_data['features'], viz_data['y_train'], 
                            viz_data['wavelengths'])
    
    plot_model_performance(viz_data['y_trues'], viz_data['y_preds'], 
                          viz_data['acc_scores'])
    
    plot_feature_importance(viz_data['model'])
    
    plot_latent_variables(viz_data['model'], viz_data['features'], 
                         viz_data['y_train'])

# 如果直接运行此脚本，执行主可视化流程
if __name__ == "__main__":
    main_visualization()