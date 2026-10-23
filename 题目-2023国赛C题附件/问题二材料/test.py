import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# 临时添加：查看附件4的所有列名
temp_loss = pd.read_excel('附件4.xlsx')
print("附件4的实际列名：", temp_loss.columns.tolist())  # 打印所有列名
