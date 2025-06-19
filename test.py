import numpy as np
import matplotlib.pyplot as plt

# 参数设置
ps_init = 18 * 30   # 初始种群数
ps_min = 4      # 最小种群数
max_evals = 300000  # 最大函数评估次数

# 生成评估次数数组
num_evals = np.linspace(0, max_evals, 500)

# 计算对应种群规模
pop_sizes = np.maximum(
    ps_min,
    np.round(ps_init - (ps_init - ps_min) * (num_evals / max_evals) ** 2)
)

i = 0
max_gen = 0
while i < max_evals:
    max_gen += 1
    n = np.round(ps_init - (ps_init - ps_min) * (num_evals / max_evals) ** 2)
    i += n
    print(i, max_gen)

# 绘制曲线
plt.figure(figsize=(8, 5))
plt.plot(num_evals, pop_sizes, color='b', linewidth=2)
plt.title('Nonlinear Quadratic Population Size Reduction', fontsize=14)
plt.xlabel('Function Evaluations (FEs)', fontsize=12)
plt.ylabel('Population Size', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(ps_min - 1, ps_init + 5)
plt.xlim(0, max_evals)
plt.tight_layout()
plt.show()
