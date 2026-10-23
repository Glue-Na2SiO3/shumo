import pandas as pd

# 读取原始测试集
input_file = '附件2处理/问题2测试集.xlsx'
output_file = '附件2处理/问题2测试集处理.xlsx'

df = pd.read_excel(input_file)
# 除第一列外所有数据除以100
cols = df.columns[1:]
df[cols] = df[cols] / 100.0
# 保存处理后的文件
df.to_excel(output_file, index=False)
print(f'处理完成，结果保存在 {output_file}')
