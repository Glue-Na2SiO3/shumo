import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False

# 创建结果保存目录
result_dir = 'result'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# --------------------------
# 1. 数据加载与预处理（匹配附件字段，区分销售/退货）
# --------------------------
def load_and_preprocess_data():
    """加载附件1-4"""
    # 读取附件（严格匹配字段）
    item_category = pd.read_excel('附件1.xlsx')[['单品编码', '分类名称']]  # 附件1
    sales = pd.read_excel('附件2.xlsx')[['销售日期', '单品编码', '销量(千克)', '销售单价(元/千克)', '销售类型']]  # 附件2
    wholesale = pd.read_excel('附件3.xlsx')[['日期', '单品编码', '批发价格(元/千克)']]  # 附件3
    loss = pd.read_excel('附件4.xlsx')[['单品编码', '损耗率(%)']]  # 附件4

    # 保存原始数据
    item_category.to_excel(os.path.join(result_dir, 'item_category_raw.xlsx'), index=False)
    sales.to_excel(os.path.join(result_dir, 'sales_raw.xlsx'), index=False)
    wholesale.to_excel(os.path.join(result_dir, 'wholesale_raw.xlsx'), index=False)
    loss.to_excel(os.path.join(result_dir, 'loss_raw.xlsx'), index=False)

    # 1.1 处理附件2：计算实际销量和销售额（退货数据本身为负，无需额外处理）
    sales['实际销量'] = sales['销量(千克)']
    sales['实际销售额'] = sales['实际销量'] * sales['销售单价(元/千克)']

    # 保存处理后的销售数据
    sales.to_excel(os.path.join(result_dir, 'sales_processed.xlsx'), index=False)

    # 1.2 关联单品与分类（通过附件1）
    sales_with_cat = pd.merge(sales, item_category, on='单品编码', how='inner')
    wholesale_with_cat = pd.merge(wholesale, item_category, on='单品编码', how='inner')
    loss_with_cat = pd.merge(loss, item_category, on='单品编码', how='inner')

    # 保存关联后的数据
    sales_with_cat.to_excel(os.path.join(result_dir, 'sales_with_cat.xlsx'), index=False)
    wholesale_with_cat.to_excel(os.path.join(result_dir, 'wholesale_with_cat.xlsx'), index=False)
    loss_with_cat.to_excel(os.path.join(result_dir, 'loss_with_cat.xlsx'), index=False)

    category_loss = loss_with_cat.groupby('分类名称')['损耗率(%)'].mean().reset_index()
    category_loss.to_excel(os.path.join(result_dir, 'category_loss.xlsx'), index=False)

    # 1.3 按品类、日期汇总实际销售数据
    category_sales = sales_with_cat.groupby(['销售日期', '分类名称']).agg(
        总销量=('实际销量', 'sum'),
        总销售额=('实际销售额', 'sum')
    ).reset_index()
    category_sales['售价'] = category_sales['总销售额'] / category_sales['总销量'] if category_sales['总销量'].sum() != 0 else 0  # 避免除零

    # 保存品类销售汇总数据
    category_sales.to_excel(os.path.join(result_dir, 'category_sales.xlsx'), index=False)

    # 1.4 按品类汇总每日批发价
    category_wholesale = wholesale_with_cat.groupby(['日期', '分类名称'])['批发价格(元/千克)'].mean().reset_index()
    category_wholesale.to_excel(os.path.join(result_dir, 'category_wholesale.xlsx'), index=False)

    # 1.5 合并数据并计算关键指标
    merged_data = pd.merge(
        category_sales,
        category_wholesale,
        left_on=['销售日期', '分类名称'],
        right_on=['日期', '分类名称'],
        how='inner'
    ).drop(columns=['日期'])

    merged_data = pd.merge(merged_data, category_loss, on='分类名称', how='inner')

    # 计算实际成本和加成率
    merged_data['实际成本'] = merged_data['批发价格(元/千克)'] / (1 - merged_data['损耗率(%)'] / 100)
    merged_data['加成率'] = (merged_data['售价'] - merged_data['实际成本']) / merged_data['实际成本'] if merged_data['实际成本'].sum() != 0 else 0  # 避免除零

    # 转换日期格式并排序
    merged_data['销售日期'] = pd.to_datetime(merged_data['销售日期'])
    merged_data = merged_data.sort_values(['分类名称', '销售日期']).reset_index(drop=True)

    # 保存合并后的数据
    merged_data.to_excel(os.path.join(result_dir, 'merged_data.xlsx'), index=False)

    return merged_data, category_loss


# --------------------------
# 2. 销量与加成率关系建模及可视化
# --------------------------
def build_demand_model(processed_data):
    """为每个品类构建销量与加成率的关系模型，并输出拟合函数及可视化"""
    categories = processed_data['分类名称'].unique()
    models = {}
    # 存储每个品类的拟合函数表达式
    fit_functions = {}
    poly = PolynomialFeatures(degree=2)  # 二次多项式拟合

    for category in categories:
        cat_data = processed_data[processed_data['分类名称'] == category].copy()
        # 过滤无效数据（总销量为0或加成率异常）
        valid_data = cat_data[(cat_data['总销量'] != 0) & (cat_data['加成率'].notna())]
        X = valid_data['加成率'].values.reshape(-1, 1)
        y = valid_data['总销量'].values

        if len(X) == 0:
            continue

        # 去除异常值（1%~99%分位数范围）
        valid_mask = (X >= np.percentile(X, 1)) & (X <= np.percentile(X, 99))
        valid_mask = valid_mask.ravel()  # 从二维数组转为一维数组
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 3:  # 数据量不足时跳过建模
            continue

        # 拟合模型
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        models[category] = {
            'model': model,
            'poly': poly
        }

        # --------------------------
        # 新增：提取系数并生成函数表达式
        # --------------------------
        # 二次多项式形式：y = a*x² + b*x + c
        coefficients = model.coef_  # 系数顺序：[a]（二次项）、[b]（一次项）
        intercept = model.intercept_  # 常数项 c
        
        # 格式化系数（保留4位小数）
        a = round(coefficients[1], 4)  # 二次项系数
        b = round(coefficients[2], 4)  # 一次项系数
        c = round(intercept, 4)        # 常数项
        
        # 构建函数表达式字符串
        function_str = f"总销量 = {a}×(加成率)² + {b}×(加成率) + {c}"
        fit_functions[category] = function_str
        
        # 打印函数表达式
        print(f"\n品类 '{category}' 的销量-加成率拟合函数：")
        print(function_str)

        # 绘制散点图和拟合曲线（在图中添加函数表达式）
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, label='实际数据', color='blue', alpha=0.6)

        # 生成拟合曲线数据
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_range_poly = poly.transform(X_range)
        y_pred = model.predict(X_range_poly)
        plt.plot(X_range, y_pred, label='拟合曲线', color='red', linewidth=2)

        # 在图中添加函数表达式文本
        plt.text(
            0.05, 0.95,  # 文本位置（相对坐标，左上角）
            function_str,
            transform=plt.gca().transAxes,
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8)
        )

        plt.xlabel('加成率')
        plt.ylabel('总销量(千克)')
        plt.title(f'{category} - 销量与加成率关系')
        plt.legend()
        plt.savefig(os.path.join(result_dir, f'{category}_demand_model.png'))
        plt.close()

    # 保存所有品类的拟合函数到文件
    with open(os.path.join(result_dir, '各品类拟合函数.txt'), 'w', encoding='utf-8') as f:
        for category, func in fit_functions.items():
            f.write(f"{category}：{func}\n")

    return models


# --------------------------
# 3. 批发价预测（SARIMAX模型，含季节性）及可视化
# --------------------------
def predict_wholesale_price(processed_data, categories, target_dates):
    """用SARIMAX预测未来批发价，并绘制预测图"""
    pred_results = {}
    seasonal_period = 365  # 年度季节性周期

    for category in categories:
        price_data = processed_data[processed_data['分类名称'] == category].copy()
        # 将数据转换为按日期索引的序列（每日一条记录）
        price_series = price_data.set_index('销售日期')['批发价格(元/千克)'].asfreq('D')  
        price_series = price_series.fillna(method='ffill')  # 填充缺失值
        
        # 关键添加：打印当前品类的历史数据量
        data_count = len(price_series.dropna())  # 排除缺失值后的有效数据量
        print(f"品类 '{category}' 的批发价历史有效数据量：{data_count} 条")

        # 如果数据量过少，提前提示
        if data_count < 30:
            print(f"警告：品类 '{category}' 数据量不足30条，可能影响预测效果！\n")

        # 简单参数搜索（数据量小，简化搜索）
        best_aic = float('inf')
        best_order = (1, 1, 1)
        best_seasonal_order = (1, 1, 1, seasonal_period)

        for p in [0, 1]:
            for d in [0, 1]:
                for q in [0, 1]:
                    for sp in [0, 1]:
                        try:
                            model = SARIMAX(
                                price_series.values,
                                order=(p, d, q),
                                seasonal_order=(sp, 1, 1, seasonal_period)
                            )
                            fit_model = model.fit(disp=False)
                            if fit_model.aic < best_aic:
                                best_aic = fit_model.aic
                                best_order = (p, d, q)
                                best_seasonal_order = (sp, 1, 1, seasonal_period)
                        except:
                            continue

        # 预测未来7天（后续代码保持不变）
        try:
            model = SARIMAX(
                price_series.values,
                order=best_order,
                seasonal_order=best_seasonal_order
            )
            fit_model = model.fit(disp=False)
            pred = fit_model.forecast(steps=len(target_dates))
            pred_results[category] = pd.Series(pred, index=target_dates)

            # 绘制预测图（后续代码保持不变）
            plt.figure(figsize=(8, 5))
            plt.plot(price_series.index, price_series.values, label='历史批发价', color='blue')
            plt.plot(target_dates, pred.values, label='预测批发价', color='red', linestyle='--')

            plt.xlabel('日期')
            plt.ylabel('批发价格(元/千克)')
            plt.title(f'{category} - 批发价预测')
            plt.legend()
            plt.savefig(os.path.join(result_dir, f'{category}_wholesale_pred.png'))
            plt.close()
        except:
            # 模型拟合失败时，用历史均值填充（后续代码保持不变）
            mean_price = price_series.mean()
            pred_results[category] = pd.Series([mean_price] * len(target_dates), index=target_dates)

            plt.figure(figsize=(8, 5))
            plt.plot(price_series.index, price_series.values, label='历史批发价', color='blue')
            plt.axhline(mean_price, label='均值填充预测', color='red', linestyle='--')

            plt.xlabel('日期')
            plt.ylabel('批发价格(元/千克)')
            plt.title(f'{category} - 批发价预测（均值填充）')
            plt.legend()
            plt.savefig(os.path.join(result_dir, f'{category}_wholesale_pred.png'))
            plt.close()

    return pred_results

# --------------------------
# 4. 收益最大化优化
# --------------------------
def optimize_revenue(models, future_wholesale, loss_rates, target_dates, processed_data):
    """求解最优补货量和加成率"""
    categories = list(models.keys())
    n_days = len(target_dates)
    n_cats = len(categories)
    loss_dict = loss_rates.set_index('分类名称')['损耗率(%)'].to_dict()

    def objective(vars):
        total_revenue = 0
        rates = vars[:n_cats * n_days].reshape(n_cats, n_days)
        补货量 = vars[n_cats * n_days:].reshape(n_cats, n_days)

        for i in range(n_days):
            for k in range(n_cats):
                category = categories[k]
                if category not in models:
                    continue  # 跳过未建模品类
                r = rates[k, i]
                x = 补货量[k, i]

                # 实际可售量计算
                实际可售量 = x * (1 - loss_dict[category] / 100)

                # 需求预测（基于模型）
                model_data = models[category]
                X_poly = model_data['poly'].transform([[r]])
                需求预测 = model_data['model'].predict(X_poly)[0]

                # 实际销量取最小
                实际销量 = min(实际可售量, 需求预测) if 需求预测 > 0 else 0

                # 收益计算
                批发价 = future_wholesale[category].iloc[i] if category in future_wholesale else 0
                实际成本 = 批发价 / (1 - loss_dict[category] / 100) if 批发价 != 0 else 0
                售价 = 实际成本 * (1 + r) if 实际成本 != 0 else 0
                total_revenue += (实际销量 * 售价) - (x * 批发价)

        return -total_revenue  # 最小化负收益

    # 变量初始化（仅对已建模品类）
    modeled_cats = [c for c in categories if c in models]
    n_modeled = len(modeled_cats)
    初始加成率 = np.array([0.3 for _ in range(n_modeled * n_days)])
    历史平均销量 = []
    for c in modeled_cats:
        cat_data = processed_data[processed_data['分类名称'] == c]
        mean_sales = cat_data['总销量'].mean() if not cat_data.empty else 0
        历史平均销量.extend([mean_sales] * n_days)
    初始补货量 = np.array(历史平均销量)
    初始变量 = np.concatenate([初始加成率, 初始补货量])

    # 约束范围
    约束范围 = [(0.1, 0.6)] * (n_modeled * n_days) + [(0, None)] * (n_modeled * n_days)

    # 求解
    result = minimize(
        objective,
        初始变量,
        method='SLSQP',
        bounds=约束范围,
        options={'maxiter': 1000}
    )

    # 整理结果
    optimal_rates = result.x[:n_modeled * n_days].reshape(n_modeled, n_days)
    optimal_补货量 = result.x[n_modeled * n_days:].reshape(n_modeled, n_days)

    优化结果 = []
    for idx, category in enumerate(modeled_cats):
        for i in range(n_days):
            批发价 = future_wholesale[category].iloc[i] if category in future_wholesale else 0
            实际成本 = 批发价 / (1 - loss_dict[category] / 100) if 批发价 != 0 else 0
            售价 = 实际成本 * (1 + optimal_rates[idx, i]) if 实际成本 != 0 else 0
            优化结果.append({
                '日期': target_dates[i].strftime('%Y-%m-%d'),
                '品类': category,
                '最优加成率': round(optimal_rates[idx, i], 4),
                '最优补货量(千克)': round(optimal_补货量[idx, i], 2),
                '预计售价(元/千克)': round(售价, 2)
            })

    return pd.DataFrame(优化结果)


# --------------------------
# 主函数执行
# --------------------------
if __name__ == "__main__":
    # 目标日期：2023年7月1日-7日
    target_dates = pd.date_range(start='2023-07-01', end='2023-07-07')

    # # 1. 数据预处理（区分销售/退货）
    # processed_data, category_loss = load_and_preprocess_data()
    # categories = processed_data['分类名称'].unique()

    processed_data, category_loss = pd.read_excel(os.path.join(result_dir, 'merged_data.xlsx')), pd.read_excel(os.path.join(result_dir, 'category_loss.xlsx'))
    categories = processed_data['分类名称'].unique()

    # # 2. 构建销量-加成率模型（仅对有效品类）并可视化
    # demand_models = build_demand_model(processed_data)

    # 3. 预测未来批发价并可视化
    future_wholesale = predict_wholesale_price(processed_data, categories, target_dates)

    # # 4. 优化求解（仅对已建模品类）
    # optimal_results = optimize_revenue(demand_models, future_wholesale, category_loss, target_dates, processed_data)

    # # 输出并保存结果
    # print("2023年7月1-7日各品类最优策略（已处理销售/退货）：")
    # print(optimal_results)
    # optimal_results.to_excel(os.path.join(result_dir, '问题2最优策略结果（含退货处理）.xlsx'), index=False)