import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ======================== 场景参数（对应题目表1，6种情形） ========================
# 参数顺序：[p1, c1, t1, p2, c2, t2, pf, a, tf, s, l, u]
# 含义：零件1次品率/单价/检测成本，零件2次品率/单价/检测成本，
#       成品装配次品率/装配成本/检测成本，成品售价/调换损失/拆解费
scenarios = [
    [0.1, 4, 2, 0.1, 18, 3, 0.1, 6, 3, 56, 6, 5],   # 情形1
    [0.2, 4, 2, 0.2, 18, 3, 0.2, 6, 3, 56, 6, 5],   # 情形2
    [0.1, 4, 2, 0.1, 18, 3, 0.1, 6, 3, 56, 30, 5],  # 情形3
    [0.2, 4, 1, 0.2, 18, 1, 0.2, 6, 2, 56, 30, 5],  # 情形4
    [0.1, 4, 8, 0.2, 18, 1, 0.1, 6, 2, 56, 10, 5],  # 情形5
    [0.05, 4, 2, 0.05, 18, 3, 0.05, 6, 3, 56, 10, 40]# 情形6
]


# ======================== 有效决策集（修正约束条件，符合文档逻辑） ========================
# 决策变量定义（严格符合题目1-10至1-13条）：
# D1：零件1是否检测（1=检测剔除不合格品；0=不检测）
# D2：零件2是否检测（1=检测剔除不合格品；0=不检测）
# D3：成品是否检测（1=检测剔除不合格品；0=不检测）
# D4：不合格品（含检测出的和用户退回的）是否拆解（1=拆解回收；0=丢弃）
# 修正：删除"D3=0则D4=0"约束，因用户退回品可拆解（1-13条），有效决策共16种
valid_decisions = []
for D1 in [0, 1]:
    for D2 in [0, 1]:
        for D3 in [0, 1]:
            for D4 in [0, 1]:
                valid_decisions.append([D1, D2, D3, D4])
valid_decisions = np.array(valid_decisions)  # 共16种有效决策


# ======================== 回收零件循环处理函数（核心修正） ========================
def process_recycled_parts(parts1, parts2, decision, scenario, max_iter=5, eps=1e-3):
    """
    处理拆解后的回收零件，循环执行检测（步骤1）和装配（步骤2），直至零件数量可忽略
    parts1/parts2：回收的零件1/2数量
    max_iter：最大循环次数，避免无限循环
    eps：最小零件数量阈值，低于此值停止循环
    """
    D1, D2, D3, D4 = decision
    p1, c1, t1, p2, c2, t2, pf, a, tf, s, l, u = scenario
    
    total_rev = 0  # 回收零件产生的累计收入
    total_cost = 0  # 回收零件产生的累计成本
    current_p1, current_p2 = parts1, parts2  # 当前可利用的回收零件数量
    
    for _ in range(max_iter):
        if current_p1 < eps and current_p2 < eps:
            break  # 零件数量过少，停止循环
        
        # 1. 回收零件检测（步骤1）
        test_cost1 = D1 * current_p1 * t1  # 零件1检测成本
        test_cost2 = D2 * current_p2 * t2  # 零件2检测成本
        total_cost += test_cost1 + test_cost2
        
        # 检测后合格零件数量（不检测则保留原始数量，含不合格品）
        ok_p1 = current_p1 * (1 - p1) if not D1 else current_p1
        ok_p2 = current_p2 * (1 - p2) if not D2 else current_p2
        
        # 2. 二次装配（步骤2）
        assemble_num = min(ok_p1, ok_p2)
        if assemble_num < eps:
            break  # 可装配数量过少，停止循环
        
        assemble_cost = assemble_num * a  # 装配成本
        total_cost += assemble_cost
        
        # 3. 二次成品质量与检测
        p1_qual = 1.0 if D1 else (1 - p1)
        p2_qual = 1.0 if D2 else (1 - p2)
        p_final_qual = p1_qual * p2_qual * (1 - pf)
        qual_num = assemble_num * p_final_qual
        defect_num = assemble_num - qual_num
        
        # 成品检测成本
        final_test_cost = D3 * assemble_num * tf
        total_cost += final_test_cost
        
        # 4. 二次销售收入
        total_rev += qual_num * s
        
        # 5. 二次不合格品处理（递归拆解）
        if D4:
            # 拆解成本
            disassemble_cost = defect_num * u
            total_cost += disassemble_cost
            # 拆解后回收的零件（进入下一轮循环）
            current_p1 = defect_num * p1_qual  # 零件1合格概率
            current_p2 = defect_num * p2_qual  # 零件2合格概率
        else:
            # 不拆解，零件直接丢弃
            current_p1, current_p2 = 0, 0
    
    return total_rev - total_cost  # 回收流程净收益


# ======================== 利润与指标计算（完整修正版） ========================
def calculate_metrics(decision, scenario, N=2000, max_recycle_iter=5):
    """
    修正逻辑：
    1. 回收零件需循环检测和装配（符合1-12条“重复步骤1和2”）
    2. 未检测成品（D3=0）的退回品可拆解（符合1-13条）
    3. 调换损失包含额外成本+合格替换品成本（符合1-26条）
    """
    D1, D2, D3, D4 = decision
    p1, c1, t1, p2, c2, t2, pf, a, tf, s, l, u = scenario

    # 1. 采购与检测成本（首次流程）
    cost_purchase = N * (c1 + c2)  # 采购N套零件
    cost_part_test = D1 * N * t1 + D2 * N * t2  # 零件检测成本

    # 2. 合格零件与装配数量
    n1_ok = N * (1 - p1) if D1 else N  # 零件1合格数量（检测剔除不合格品）
    n2_ok = N * (1 - p2) if D2 else N  # 零件2合格数量
    assemble_num = min(n1_ok, n2_ok)    # 首次装配数量
    cost_assemble = assemble_num * a    # 装配成本

    # 3. 首次成品质量（1-3条：双合格才可能合格，存在装配次品率）
    p1_qual = 1.0 if D1 else (1 - p1)
    p2_qual = 1.0 if D2 else (1 - p2)
    p_final_qual = p1_qual * p2_qual * (1 - pf)
    qual_num = assemble_num * p_final_qual        # 合格成品数量
    defect_num = assemble_num - qual_num          # 不合格成品数量

    # 4. 成品检测成本
    cost_final_test = D3 * assemble_num * tf

    # 5. 不合格品处理（分检测/未检测两种场景）
    cost_disassemble = 0
    recycle_revenue = 0
    cost_exchange = 0
    replace_cost = 0  # 替换合格产品的成本（符合1-26条“调换损失不含产品本身”）

    if D3:
        # 场景1：成品检测→不合格品未流向市场
        if D4:
            # 拆解回收（1-12条）
            cost_disassemble = defect_num * u
            # 回收零件进入循环处理
            recycle_revenue = process_recycled_parts(
                parts1=defect_num * p1_qual,
                parts2=defect_num * p2_qual,
                decision=decision,
                scenario=scenario,
                max_iter=max_recycle_iter
            )
    else:
        # 场景2：成品未检测→不合格品流向市场（1-13条）
        cost_exchange = defect_num * l  # 调换损失（额外成本）
        # 需向用户提供合格替换品，其成本等同于合格成品的生产成本
        replace_qual_num = defect_num  # 假设1:1替换
        replace_cost = replace_qual_num * (
            (c1 + c2) +  # 替换品零件成本
            (D1*t1 + D2*t2) +  # 替换品检测成本
            a +  # 替换品装配成本
            D3*tf  # 替换品检测成本
        )
        # 用户退回的不合格品处理
        if D4:
            cost_disassemble = defect_num * u  # 退回品拆解成本
            # 回收零件进入循环处理
            recycle_revenue = process_recycled_parts(
                parts1=defect_num * p1_qual,
                parts2=defect_num * p2_qual,
                decision=decision,
                scenario=scenario,
                max_iter=max_recycle_iter
            )

    # 6. 总收入与总成本
    revenue = qual_num * s  # 首次销售收益
    total_cost = (cost_purchase + cost_part_test + cost_assemble + 
                 cost_final_test + cost_disassemble + cost_exchange + replace_cost)
    profit = revenue + recycle_revenue - total_cost  # 总利润=首次收益+回收收益-总成本

    # 返回中间指标（用于可视化）
    return {
        'profit': profit,
        'costs': {
            '采购': cost_purchase,
            '零件检测': cost_part_test,
            '装配': cost_assemble,
            '成品检测': cost_final_test,
            '拆解': cost_disassemble,
            '调换损失': cost_exchange,
            '替换品成本': replace_cost
        },
        '数量': {
            '零件1合格': n1_ok,
            '零件2合格': n2_ok,
            '装配总数': assemble_num,
            '合格成品': qual_num,
            '不合格成品': defect_num
        },
        '回收收益': recycle_revenue,
        '决策': decision
    }


# ======================== 单场景分析与可视化 ========================
def analyze_scenario(scenario_idx):
    scenario = scenarios[scenario_idx]
    case_num = scenario_idx + 1
    metrics_list = [calculate_metrics(dec, scenario) for dec in valid_decisions]
    
    # 筛选最优决策（利润最大）
    best_idx = np.argmax([m['profit'] for m in metrics_list])
    best_metrics = metrics_list[best_idx]
    best_dec = best_metrics['决策']

    # 构建可视化布局（3行2列）
    fig = plt.figure(figsize=(15, 18))
    gs = GridSpec(3, 2, figure=fig)
    fig.suptitle(f'情形{case_num} 最优决策：{best_dec}', fontsize=16, fontweight='bold')

    # 1. 所有决策的利润分布
    ax1 = fig.add_subplot(gs[0, 0])
    profits = [m['profit'] for m in metrics_list]
    ax1.boxplot(profits, patch_artist=True, boxprops=dict(facecolor='#E6F3FF'))
    ax1.scatter(np.ones(len(profits)), profits, c='#FF4D4F', alpha=0.6, label='各决策利润')
    ax1.axhline(best_metrics['profit'], color='#73C0DE', linestyle='--', 
                label=f'最优利润：{best_metrics["profit"]:.1f}元')
    ax1.set_title('所有决策的利润分布', fontsize=12)
    ax1.legend()

    # 2. 最优决策的成本构成
    ax2 = fig.add_subplot(gs[0, 1])
    costs = best_metrics['costs']
    labels = list(costs.keys())
    values = list(costs.values())
    ax2.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=sns.color_palette('pastel'), wedgeprops=dict(width=0.3, edgecolor='white'))
    ax2.set_title('最优决策的成本构成', fontsize=12)

    # 3. 零件合格数与装配数对比
    ax3 = fig.add_subplot(gs[1, 0])
    qty = best_metrics['数量']
    data = [qty['零件1合格'], qty['零件2合格'], -abs(qty['零件1合格']-qty['零件2合格']), qty['装配总数']]
    labels = ['零件1合格', '零件2合格', '数量差', '实际装配']
    colors = ['#91CC75', '#FAC858', '#FF6B6B', '#6699CC']
    sns.barplot(x=labels, y=data, palette=colors, ax=ax3)
    ax3.set_ylabel('数量（件）')
    ax3.set_title('零件合格数与装配数关系', fontsize=12)

    # 4. 成品质量分布
    ax4 = fig.add_subplot(gs[1, 1])
    qual = [qty['合格成品']]
    defect = [qty['不合格成品']]
    ax4.bar(['成品'], qual, label='合格', color='#52C41A')
    ax4.bar(['成品'], defect, bottom=qual, label='不合格', color='#FF4D4F')
    ax4.set_ylabel('数量（件）')
    ax4.set_title('成品质量分布', fontsize=12)
    ax4.legend()

    # 5. D1-D2决策的利润敏感性
    ax5 = fig.add_subplot(gs[2, 0])
    profit_matrix = np.zeros((2, 2))
    best_D3, best_D4 = best_dec[2], best_dec[3]
    for d1 in [0, 1]:
        for d2 in [0, 1]:
            dec = [d1, d2, best_D3, best_D4]
            idx = np.where((valid_decisions == dec).all(axis=1))[0][0]
            profit_matrix[d1, d2] = metrics_list[idx]['profit']
    sns.heatmap(profit_matrix, annot=True, fmt='.1f', cmap='RdBu_r', ax=ax5,
                xticklabels=['不检测', '检测'], yticklabels=['不检测', '检测'],
                cbar_kws={'label': '利润（元）'})
    ax5.set_xlabel('零件2检测决策')
    ax5.set_ylabel('零件1检测决策')
    ax5.set_title('D1-D2决策的利润敏感性', fontsize=12)

    # 6. 最优 vs 最差决策成本对比
    ax6 = fig.add_subplot(gs[2, 1], polar=True)
    worst_idx = np.argmin([m['profit'] for m in metrics_list])
    worst_metrics = metrics_list[worst_idx]
    cost_types = list(costs.keys())
    num_vars = len(cost_types)
    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    best_vals = [costs[k] for k in cost_types]
    worst_vals = [worst_metrics['costs'][k] for k in cost_types]
    best_vals += best_vals[:1]
    worst_vals += worst_vals[:1]
    angles += angles[:1]
    ax6.fill(angles, best_vals, color='#6699CC', alpha=0.4, label='最优决策')
    ax6.fill(angles, worst_vals, color='#FF6B6B', alpha=0.4, label='最差决策')
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(cost_types)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax6.set_title('成本结构对比（雷达图）', fontsize=12, pad=20)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'情形{case_num}_决策分析.png', dpi=300, bbox_inches='tight')
    plt.close()

    return best_dec, best_metrics['profit']


# ======================== 主程序 ========================
if __name__ == "__main__":
    all_best_decisions = []
    
    # 逐场景分析
    for i in range(6):
        best_dec, best_profit = analyze_scenario(i)
        all_best_decisions.append(best_dec)
        print(f"情形{i+1} 最优决策：{best_dec}，最大利润：{best_profit:.1f}元")

    # 全局决策矩阵可视化
    plt.figure(figsize=(10, 6))
    dec_matrix = np.array(all_best_decisions)
    rows, cols = dec_matrix.shape
    cell_size = 0.8

    color_map = {0: '#E0F7FA', 1: '#FF7043'}
    edge_color = '#444444'

    for i in range(rows):
        for j in range(cols):
            rect = plt.Rectangle(
                (j - cell_size/2, i - cell_size/2), 
                cell_size, cell_size, 
                facecolor=color_map[dec_matrix[i, j]], 
                edgecolor=edge_color, linewidth=1.2
            )
            plt.gca().add_patch(rect)
            plt.text(j, i, str(dec_matrix[i, j]), 
                     ha='center', va='center', fontsize=14, fontweight='bold', color='#222')

    plt.xlim(-0.5, cols - 0.5)
    plt.ylim(-0.5, rows - 0.5)
    plt.xticks(range(cols), [
        "零件1检测", "零件2检测", 
        "成品检测", "不合格品拆解"
    ], fontsize=11, rotation=15)
    plt.yticks(range(rows), [f"情形{i+1}" for i in range(rows)], fontsize=11)
    plt.gca().set_aspect('equal')
    plt.title("各情形最优决策矩阵", fontsize=14, pad=25)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("全局决策矩阵.png", dpi=300, bbox_inches='tight')
    plt.show()