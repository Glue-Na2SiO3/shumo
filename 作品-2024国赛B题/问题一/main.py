from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']


# ========== 参数定义（核心逻辑纠正） ==========
p0 = 0.1       # 标称合格次品率（H0边界）
p1 = 0.15      # 参考超标次品率（1.5倍标称值，工业临界值）
alpha = 0.10   # 错拒概率上限（H0为真时拒绝，≤10%）
beta = 0.05    # 错收概率上限（H1为真时接受，≤5%）
accept_p0_min = 1 - alpha  # 合格批接收概率≥90%
reject_p1_min = 1 - beta   # 超标批拒收概率≥95%

# ========== 枚举求解最优(n,c) ==========
n_min = None
c_best = None
debug_log = []  # 记录中间验证数据（前20组有效解）

# 扩大n范围以覆盖理论最优
for n in range(1, 400):  
    for c in range(0, n+1):  # 从0开始遍历临界值c
        # 计算合格批接收概率（d < c 的概率）
        if c == 0:
            accept_p0 = 0.0  # c=0时，d<0不可能，接收概率为0
        else:
            accept_p0 = binom.cdf(c-1, n, p0)
        
        # 计算超标批拒收概率（d ≥ c 的概率）
        if c == 0:
            reject_p1 = 1.0  # c=0时，d≥0必然，拒收概率为1
        else:
            reject_p1 = 1 - binom.cdf(c-1, n, p1)
        
        # 验证双约束：合格批接收≥90%，超标批拒收≥95%
        if (accept_p0 >= accept_p0_min) and (reject_p1 >= reject_p1_min):
            # 记录最小n和对应c（首次找到的c为当前n的最小值）
            if n_min is None or n < n_min:
                n_min = n
                c_best = c
                # 记录中间数据（调试用）
                if len(debug_log) < 20:
                    debug_log.append({
                        "n": n,
                        "c": c,
                        "accept_p0": accept_p0,
                        "reject_p1": reject_p1
                    })
            break  # 跳出当前n的c循环，继续下一个n

# ========== 结果输出与验证 ==========
if n_min is not None and c_best is not None:
    print("\n===== 最优抽样方案 ======")
    print(f"样本量 n = {n_min}")
    print(f"临界值 c = {c_best}")
    # 计算并输出关键概率
    accept_p0 = binom.cdf(c_best-1, n_min, p0) if c_best > 0 else 0.0
    reject_p1 = 1 - binom.cdf(c_best-1, n_min, p1) if c_best > 0 else 1.0
    print(f"p={p0}时接收概率：{accept_p0:.4f}（需≥{accept_p0_min:.2f}）")
    print(f"p={p1}时拒收概率：{reject_p1:.4f}（需≥{reject_p1_min:.2f}）")
    print(f"p={p0}时错拒概率：{1 - accept_p0:.4f}（需≤{alpha:.2f}）")
    print(f"p={p1}时错收概率：{1 - reject_p1:.4f}（需≤{beta:.2f}）")

    # 输出中间验证数据（前10组）
    print("\n===== 中间验证数据（前10条） ======")
    for i, info in enumerate(debug_log[:10]):
        print(f"第{i+1}组：n={info['n']}, c={info['c']}")
        print(f"  合格批接收概率：{info['accept_p0']:.4f}, 超标批拒收概率：{info['reject_p1']:.4f}")

    # ========== 绘制OC曲线（操作特性曲线） ==========
    p_range = np.linspace(0, 0.3, 200)
    accept_probs = []
    for p in p_range:
        if c_best == 0:
            prob = 0.0
        else:
            prob = binom.cdf(c_best-1, n_min, p)
        accept_probs.append(prob)

    plt.figure(figsize=(8, 5))
    plt.plot(p_range, accept_probs, 'b-', linewidth=2, label=f'OC曲线 (n={n_min}, c={c_best})')
    plt.axvline(p0, color='r', linestyle='--', label=f'标称值 p0={p0}')
    plt.axvline(p1, color='purple', linestyle='--', label=f'参考超标值 p1={p1}')
    plt.axhline(accept_p0_min, color='g', linestyle='--', label=f'合格批接收置信度 {accept_p0_min:.0%}')
    plt.axhline(reject_p1_min, color='orange', linestyle='--', label=f'超标批拒收置信度 {reject_p1_min:.0%}')
    plt.xlabel('实际次品率 p')
    plt.ylabel('接收概率')
    plt.title('抽样方案操作特性曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig("OC_curve.png", bbox_inches='tight')
    plt.show()
else:
    print("未找到满足条件的抽样方案，请检查参数或扩大n的范围（建议n≤3000）。")