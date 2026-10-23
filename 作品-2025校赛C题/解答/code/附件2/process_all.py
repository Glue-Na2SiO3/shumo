import subprocess
import sys
import os

scripts = [
    'model2_PLS-DA.py',
    'model2_PCA+逻辑回归.py',
    'model2_LightGBM.py',
    'model2_SVM.py',
    'model2_Stacking集成.py',
    'model2_XGBoost增量学习.py',
    'model2_VAE降维.py',
    'model2_随机森林+RFE.py',
    # 'compare_model2.py'
]

for script in scripts:
    print(f'正在运行: {script}')
    result = subprocess.run([sys.executable, script], cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        print(f'运行 {script} 时出错，已中断')
        break
    print(f'{script} 运行完成\n')
print('全部模型运行结束')
