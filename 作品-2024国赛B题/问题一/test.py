from scipy.stats import binom

n = 379
c = 46
p0 = 0.10
p1 = 0.15

# 合格批接收概率
prob_accept_p0 = binom.cdf(c - 1, n, p0)

# 超标批拒收概率
prob_reject_p1 = 1 - binom.cdf(c - 1, n, p1)

print(f"合格批接收概率：{prob_accept_p0:.4f}")
print(f"超标批拒收概率：{prob_reject_p1:.4f}")