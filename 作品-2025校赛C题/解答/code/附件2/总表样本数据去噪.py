import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import random

# 对每个样本（行）进行归一化和降噪，输出格式与输入一致
def denoise_table(input_file, output_file, k=0.1):
    df = pd.read_excel(input_file)
    print('开始归一化...')
    # 归一化
    df.iloc[:, 1:] = df.iloc[:, 1:] / 100.0
    print('归一化完成，开始clip异常值...')
    spectral_columns = df.columns[1:]
    # clip异常值
    df[spectral_columns] = df[spectral_columns].clip(lower=0, upper=1)
    print('clip完成，开始逐行SG滤波...')
    # SG滤波
    for i in range(df.shape[0]):
        spectrum = df.iloc[i, 1:].values
        wl = min(15, len(spectrum))
        if wl % 2 == 0:
            wl -= 1
        if wl < 3:
            wl = 3
        smoothed = savgol_filter(spectrum, window_length=wl, polyorder=2, mode='nearest')
        # 二次clip
        smoothed = np.clip(smoothed, 0, 1)
        df.iloc[i, 1:] = smoothed
        if (i+1) % 10 == 0 or i == df.shape[0]-1:
            print(f'已处理 {i+1}/{df.shape[0]} 个样本')
    print('全部样本SG滤波完成，正在保存...')
    df.to_excel(output_file, index=False)
    print(f'保存完成：{output_file}')

def visualize_denoising(input_file, output_file, num_samples=3):
    print('开始可视化对比...')
    df_raw = pd.read_excel(input_file)
    df_denoised = pd.read_excel(output_file)
    wavelengths = df_raw.columns[1:]
    df_raw.iloc[:, 1:] = df_raw.iloc[:, 1:] / 100.0
    sample_indices = random.sample(range(df_raw.shape[0]), num_samples)
    plt.figure(figsize=(12, 4 * num_samples))
    for i, idx in enumerate(sample_indices):
        plt.subplot(num_samples, 1, i+1)
        plt.plot(wavelengths, df_raw.iloc[idx, 1:], label='Raw (normalized)', alpha=0.7)
        plt.plot(wavelengths, df_denoised.iloc[idx, 1:], label='Processed', alpha=0.7)
        plt.title(f'Sample {idx+1} Spectrum Comparison')
        plt.xlabel('Wavelength')
        plt.ylabel('Reflectance')
        plt.ylim(0, 1.1)
        plt.legend()
    plt.tight_layout()
    plt.show()
    print('可视化完成')

# === 读取和处理数据 ===
input_file = '附件2处理/总表.xlsx'
output_file = '附件2处理/总表降噪.xlsx'
denoise_table(input_file, output_file)
visualize_denoising(input_file, output_file)
print("已完成！结果保存在 附件2处理/总表降噪.xlsx")
