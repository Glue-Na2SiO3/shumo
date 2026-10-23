import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table
import matplotlib.colors as mcolors
import matplotlib.patches as patches



def create_classification_table(input_excel, output_image):
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 微软雅黑和黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 读取Excel数据
    df = pd.read_excel(input_excel)


    # 定义颜色
    colors = {
        'header': '#546e7a',  # 岩灰
        'id_bg': '#f8f9fa',   # ID背景
        'label0': '#F4F8D3',  # 淡蓝
        'label1': '#e8f5e9',  # 浅绿
        'label2': '#d7e6d3'   # 淡紫
    }

    # 将数据分组，每组20个样本
    groups = []
    for i in range(0, len(df), 20):
        group = df.iloc[i:i+20]
        groups.append(group)

    # 创建图形
    fig, ax = plt.subplots(figsize=(20, len(groups)*1.5 + 1))
    ax.axis('off')

    # 创建表格
    table = Table(ax, bbox=[0, 0, 1, 1])

    row_idx = 0
    for group_idx, group in enumerate(groups):
        # ID行
        table.add_cell(row_idx, 0, width=0.1, height=0.1, text='待分类ID',
                      loc='center', facecolor=colors['header'],
                      edgecolor='white')
        cell = table[(row_idx, 0)]
        cell.get_text().set_color('white')

        for i, (_, row) in enumerate(group.iterrows()):
            id_value = row['ID']
            try:
                id_text = str(int(id_value))
            except (ValueError, TypeError):
                id_text = str(id_value)
            table.add_cell(row_idx, i+1, width=0.1, height=0.1,
                          text=id_text, loc='center',
                          facecolor=colors['id_bg'], edgecolor='white')

        row_idx += 1

        # 标签行
        table.add_cell(row_idx, 0, width=0.1, height=0.1, text='预测标签',
                      loc='center', facecolor=colors['header'],
                      edgecolor='white')
        cell = table[(row_idx, 0)]
        cell.get_text().set_color('white')

        for i, (_, row) in enumerate(group.iterrows()):
            label = str(int(row['类别']))
            facecolor = colors[f'label{label}']
            table.add_cell(row_idx, i+1, width=0.1, height=0.1,
                          text=label, loc='center',
                          facecolor=facecolor, edgecolor='white')

        row_idx += 1

    # 添加表格到图形
    ax.add_table(table)

    # 保存图片
    plt.savefig(output_image, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"表格已保存为: {output_image}")

# 使用示例
input_excel = '问题4测试集2结果.xlsx'  # 替换为你的Excel文件路径
output_image = '问题4测试集2结果.png'  # 输出图片路径
create_classification_table(input_excel, output_image)