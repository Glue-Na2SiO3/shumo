import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ======================== 题目参数（严格对应表2） ========================
# 零配件参数：{编号: [次品率, 购买单价, 检测成本]}
parts_params = {
    1: [0.1, 2, 1],   2: [0.1, 8, 1],   3: [0.1, 12, 2],
    4: [0.1, 2, 1],   5: [0.1, 8, 1],   6: [0.1, 12, 2],
    7: [0.1, 8, 1],   8: [0.1, 12, 2]
}

# 半成品参数：{编号: [组成零件, 次品率, 装配成本, 检测成本, 拆解费用]}
semi_params = {
    1: ([1, 2, 3], 0.1, 8, 4, 6),
    2: ([4, 5, 6], 0.1, 8, 4, 6),
    3: ([7, 8], 0.1, 8, 4, 6)
}

# 成品参数（对应表2和问题3描述）
final_params = {
    'components': [1, 2, 3], 'p': 0.1, 'assemble_cost': 8,
    'test_cost': 6, 'disassemble_cost': 10, 'price': 200, 'exchange_loss': 40
}

N = 15000  # 初始采购量（每种零配件数量）


# ======================== 生成所有决策方案 ========================
def generate_all_decisions():
    """生成8192种决策：[零件8个, 半成品3个, 成品1个, 拆解1个]"""
    decisions = []
    for d1 in [0, 1]:
        for d2 in [0, 1]:
            for d3 in [0, 1]:
                for d4 in [0, 1]:
                    for d5 in [0, 1]:
                        for d6 in [0, 1]:
                            for d7 in [0, 1]:
                                for d8 in [0, 1]:
                                    part_dec = [d1, d2, d3, d4, d5, d6, d7, d8]
                                    for s1 in [0, 1]:
                                        for s2 in [0, 1]:
                                            for s3 in [0, 1]:
                                                for f in [0, 1]:
                                                    for dis in [0, 1]:
                                                        decisions.append({
                                                            'part': part_dec,
                                                            'semi': [s1, s2, s3],
                                                            'final': f,
                                                            'dis': dis,
                                                            'id': len(decisions)  # 决策唯一标识
                                                        })
    return decisions

all_decisions = generate_all_decisions()
print(f"生成决策方案总数：{len(all_decisions)}种")


# ======================== 利润计算函数（返回详细收支） ========================
def calculate_profit(dec):
    """计算利润并返回各阶段收支明细"""
    part_dec = dec['part']
    semi_dec = dec['semi']
    final_dec = dec['final']
    dis_dec = dec['dis']
    
    # 1. 零件阶段
    cost_purchase = sum(N * parts_params[p+1][1] for p in range(8))  # 采购成本
    cost_part_test = 0
    valid_parts = []
    for p in range(8):
        p_id = p + 1
        p_defect, _, p_test = parts_params[p_id]
        if part_dec[p] == 1:
            cost_part_test += N * p_test  # 检测成本
            valid_parts.append(N * (1 - p_defect))
        else:
            valid_parts.append(N * (1 - p_defect))
    
    # 2. 半成品阶段
    cost_semi_assem = 0  # 半成品装配成本
    cost_semi_test = 0   # 半成品检测成本
    valid_semi = []
    for s in range(3):
        s_id = s + 1
        comps, s_defect, s_assem, s_test, _ = semi_params[s_id]
        comp_indices = [p-1 for p in comps]
        assemble_qty = min(valid_parts[p] for p in comp_indices)
        cost_semi_assem += assemble_qty * s_assem  # 装配成本
        if semi_dec[s] == 1:
            cost_semi_test += assemble_qty * s_test  # 检测成本
            valid_semi.append(assemble_qty * (1 - s_defect))
        else:
            valid_semi.append(assemble_qty * (1 - s_defect))
    cost_semi_total = cost_semi_assem + cost_semi_test
    
    # 3. 成品阶段
    final_assemble = min(valid_semi)
    cost_final_assem = final_assemble * final_params['assemble_cost']  # 成品装配成本
    cost_final_test = final_assemble * final_params['test_cost'] if final_dec == 1 else 0  # 成品检测成本
    cost_final_total = cost_final_assem + cost_final_test
    
    # 合格成品与收入
    final_valid = final_assemble * (1 - final_params['p'])
    revenue_final = final_valid * final_params['price']
    final_defect = final_assemble - final_valid
    
    # 4. 调换损失
    cost_exchange = (1 - final_dec) * final_defect * final_params['exchange_loss']
    
    # 5. 拆解成本与回收
    cost_disassemble = final_defect * final_params['disassemble_cost'] if dis_dec == 1 else 0
    recycled_semi = [final_defect, final_defect, final_defect] if dis_dec == 1 else [0, 0, 0]
    
    # 6. 回收循环收益（记录各轮数据）
    revenue_recycle = 0
    recycle_rounds = []  # 存储各轮回收收益
    current_semi = recycled_semi.copy()
    for round_idx in range(3):
        if sum(current_semi) < 1:
            break
        round_revenue = 0
        round_cost = 0
        
        # 回收半成品检测与合格数量
        valid_recy_semi = []
        for s in range(3):
            qty = current_semi[s]
            s_id = s + 1
            s_defect, s_test = semi_params[s_id][1], semi_params[s_id][3]
            if semi_dec[s] == 1:
                round_cost += qty * s_test  # 检测成本
                valid = qty * (1 - s_defect)
                valid_recy_semi.append(valid)
            else:
                valid_recy_semi.append(qty * (1 - s_defect))
        
        # 装配新成品
        new_final_assemble = min(valid_recy_semi)
        round_cost += new_final_assemble * final_params['assemble_cost']  # 装配成本
        if final_dec == 1:
            round_cost += new_final_assemble * final_params['test_cost']  # 检测成本
        
        # 新成品收入
        new_valid = new_final_assemble * (1 - final_params['p'])
        round_revenue = new_valid * final_params['price']
        new_defect = new_final_assemble - new_valid
        
        # 累计收益与下一轮回收
        revenue_recycle += (round_revenue - round_cost)
        recycle_rounds.append({
            'round': round_idx + 1,
            'revenue': round_revenue,
            'cost': round_cost,
            'net': round_revenue - round_cost
        })
        current_semi = [new_defect, new_defect, new_defect] if dis_dec == 1 else [0, 0, 0]
    
    # 总收支计算
    total_cost = cost_purchase + cost_part_test + cost_semi_total + cost_final_total + cost_exchange + cost_disassemble
    total_revenue = revenue_final + revenue_recycle
    profit = total_revenue - total_cost
    
    # 返回详细结果（用于可视化）
    return {
        'profit': profit,
        'revenue': {'final': revenue_final, 'recycle': revenue_recycle, 'total': total_revenue},
        'cost': {
            'purchase': cost_purchase,
            'part_test': cost_part_test,
            'semi_assem': cost_semi_assem,
            'semi_test': cost_semi_test,
            'final_assem': cost_final_assem,
            'final_test': cost_final_test,
            'exchange': cost_exchange,
            'disassemble': cost_disassemble,
            'total': total_cost
        },
        'recycle_rounds': recycle_rounds,
        'decision': dec
    }


# ======================== 全枚举求解与结果分析 ========================
if __name__ == "__main__":
    print("开始全枚举计算（8192种决策）...")
    results = []
    for i, dec in enumerate(all_decisions):
        if i % 1000 == 0:
            print(f"已计算 {i}/{len(all_decisions)} 种决策")
        results.append(calculate_profit(dec))
    
    # 提取最优决策
    best_idx = np.argmax([r['profit'] for r in results])
    best_result = results[best_idx]
    best_dec = best_result['decision']
    best_profit = best_result['profit']

    # -------------------------- 输出最优决策 --------------------------
    print("\n" + "="*50)
    print("问题3最优决策方案")
    print("="*50)
    print("1. 零配件检测决策（1=检测，0=不检测）：")
    for i in range(8):
        print(f"   零配件{i+1}：{best_dec['part'][i]}")
    print("\n2. 半成品检测决策：")
    for i in range(3):
        print(f"   半成品{i+1}（组成零件：{semi_params[i+1][0]}）：{best_dec['semi'][i]}")
    print(f"\n3. 成品检测决策：{'检测' if best_dec['final'] == 1 else '不检测'}")
    print(f"4. 不合格品拆解决策：{'拆解' if best_dec['dis'] == 1 else '不拆解'}")
    print(f"\n最大利润：{best_profit:.2f}元")
    print("="*50 + "\n")


    # # -------------------------- 可视化中间结果 --------------------------
    # # 1. 利润分布直方图（优化版）
    # plt.figure(figsize=(12, 6))
    # profits = [r['profit'] for r in results]
    # sns.histplot(profits, bins=30, kde=True, color='#4CAF50', edgecolor='white')
    # plt.axvline(best_profit, color='#FF5722', linestyle='--', linewidth=2,
    #             label=f'最优利润：{best_profit:.2f}元')
    # plt.xlabel('利润（元）', fontsize=12)
    # plt.ylabel('决策数量', fontsize=12)
    # plt.title('所有决策方案的利润分布', fontsize=14, pad=20)
    # plt.legend(fontsize=11)
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.savefig('利润分布.png', dpi=300, bbox_inches='tight')
    # plt.show()


    # # 2. 最优决策的成本构成（堆叠条形图）
    # cost_data = best_result['cost']
    # cost_labels = ['采购成本', '零件检测', '半成品装配', '半成品检测', 
    #                '成品装配', '成品检测', '调换损失', '拆解费用']
    # cost_values = [
    #     cost_data['purchase'], cost_data['part_test'],
    #     cost_data['semi_assem'], cost_data['semi_test'],
    #     cost_data['final_assem'], cost_data['final_test'],
    #     cost_data['exchange'], cost_data['disassemble']
    # ]

    # plt.figure(figsize=(10, 6))
    # bars = plt.bar(cost_labels, cost_values, color=sns.color_palette('pastel', 8))
    # for bar in bars:
    #     height = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width()/2., height + 5000,
    #              f'{height:.0f}', ha='center', fontsize=10)
    # plt.ylabel('成本金额（元）', fontsize=12)
    # plt.title('最优决策的成本构成', fontsize=14, pad=20)
    # plt.xticks(rotation=30, ha='right')
    # plt.grid(axis='y', alpha=0.3)
    # plt.tight_layout()
    # plt.savefig('最优决策成本构成.png', dpi=300, bbox_inches='tight')
    # plt.show()


    # # 3. 最优决策的收入构成（饼图）
    # revenue_data = best_result['revenue']
    # revenue_labels = ['成品销售收入', '回收循环收益']
    # revenue_values = [revenue_data['final'], revenue_data['recycle']]

    # plt.figure(figsize=(8, 8))
    # wedges, texts, autotexts = plt.pie(
    #     revenue_values, labels=revenue_labels, autopct='%1.1f%%',
    #     startangle=90, colors=['#2196F3', '#FFC107'],
    #     textprops={'fontsize': 12}
    # )
    # plt.setp(autotexts, size=12, weight='bold')
    # plt.title('最优决策的收入构成', fontsize=14, pad=20)
    # plt.tight_layout()
    # plt.savefig('最优决策收入构成.png', dpi=300, bbox_inches='tight')
    # plt.show()


    # # 4. 回收循环收益分析（折线图）
    # recycle_data = best_result['recycle_rounds']
    # if recycle_data:
    #     rounds = [d['round'] for d in recycle_data]
    #     net_profit = [d['net'] for d in recycle_data]

    #     plt.figure(figsize=(10, 6))
    #     plt.plot(rounds, net_profit, marker='o', color='#9C27B0', linewidth=2, markersize=8)
    #     for r, np in zip(rounds, net_profit):
    #         plt.text(r, np + 1000, f'{np:.0f}元', ha='center', fontsize=10)
    #     plt.xlabel('回收循环轮次', fontsize=12)
    #     plt.ylabel('净收益（元）', fontsize=12)
    #     plt.title('拆解回收各轮次净收益', fontsize=14, pad=20)
    #     plt.grid(alpha=0.3)
    #     plt.xticks(rounds)
    #     plt.tight_layout()
    #     plt.savefig('回收循环收益.png', dpi=300, bbox_inches='tight')
    #     plt.show()


    # # 5. 决策维度分布热力图（检测决策分布）
    # # 提取所有决策的检测维度（零件/半成品/成品）
    # part_dec_count = np.zeros((8, 2))  # 零件检测决策分布（[不检测, 检测]）
    # semi_dec_count = np.zeros((3, 2))  # 半成品检测决策分布
    # final_dec_count = np.zeros(2)      # 成品检测决策分布

    # for r in results:
    #     dec = r['decision']
    #     # 零件决策
    #     for i in range(8):
    #         part_dec_count[i, dec['part'][i]] += 1
    #     # 半成品决策
    #     for i in range(3):
    #         semi_dec_count[i, dec['semi'][i]] += 1
    #     # 成品决策
    #     final_dec_count[dec['final']] += 1

    # # 绘制零件检测决策热力图
    # plt.figure(figsize=(10, 5))
    # sns.heatmap(part_dec_count, annot=True, fmt='.0f', cmap='Blues', 
    #             xticklabels=['不检测', '检测'], 
    #             yticklabels=[f'零件{i+1}' for i in range(8)])
    # plt.title('所有决策的零件检测分布', fontsize=14, pad=20)
    # plt.tight_layout()
    # plt.savefig('零件检测决策分布.png', dpi=300, bbox_inches='tight')
    # plt.show()