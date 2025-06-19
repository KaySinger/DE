from multiprocessing import Pool, cpu_count
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from numba import jit
from scipy.integrate import odeint

from LSHADE_py import LSHADE
from LSHADE_cnEpsin_py import LSHADE_cnEpsin
from jSO_py import jSO
import cec2017.functions as functions

# 正态分布模拟，得到的结果用于物质稳态浓度
def simulate_normal_distribution(mu, sigma, total_concentration, x_values, scale_factor):
    concentrations = np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    concentrations /= sum(concentrations)
    concentrations *= scale_factor
    return concentrations

# 定义非线性微分方程组
@jit(nopython=True)
def equations1(p, t, k_values):
    dpdt = np.zeros_like(p)
    k = k_values[:40]
    k_inv = k_values[40:]
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] + k_inv[0] * p[2] - k[1] * p[1] ** 2
    for i in range(2, 40):
        dpdt[i] = k[i - 1] * p[i - 1] ** 2 + k_inv[i - 1] * p[i + 1] - k_inv[i - 2] * p[i] - k[i] * p[i] ** 2
    dpdt[40] = k[39] * p[39] ** 2 - k_inv[38] * p[40]
    return dpdt

# 定义非线性微分方程组
@jit(nopython=True)
def equations2(p, t, k_values):
    dpdt = np.zeros_like(p)
    k = k_values[:10]
    dpdt[0] = - k[0] * p[0]
    dpdt[1] = k[0] * p[0] - k[1] * p[1] ** 2
    for i in range(2, 10):
        dpdt[i] = k[i - 1] * p[i - 1] ** 2 - k[i] * p[i] ** 2
    dpdt[10] = k[9] * p[9] ** 2
    return dpdt

# 定义目标函数
def objective(k):
    # 正向系数递增性惩罚项
    # k_forward = k[1:40]
    # penalty = 0.0
    # # 计算所有相邻k的递减量，若k[i+1] < k[i]则施加惩罚
    # for i in range(len(k_forward) - 1):
    #     if k_forward[i + 1] < k_forward[i]:
    #         penalty += (k_forward[i] - k_forward[i + 1]) ** 2  # 平方惩罚项
    # penalty_weight = 1e6  # 惩罚权重（根据问题规模调整）
    # total_penalty = penalty_weight * penalty
    initial_p = [10.0] + [0] * 10
    t = np.linspace(0, 1000, 1000)
    # 求解微分方程
    sol = odeint(equations2, initial_p, t, args=(k,))
    # 选取t>=900时的所有解（假设t=1000时有1000个点，索引900对应t=900）
    selected_sol = sol[-1, :]
    # 理想浓度
    ideal_p = np.array([0] + list(target_p))
    # 计算所有选中时间点的误差平方和
    sum_error = np.sum((selected_sol - ideal_p) ** 2)

    return sum_error

# 求得理想最终浓度
target_p = simulate_normal_distribution(mu=5.5, sigma=10, total_concentration=1.0, x_values=np.arange(1, 11),
                                            scale_factor=10.0)
x_values = [f'P{i}' for i in range(1, 11)]  # 定义图像横坐标
print("理想最终浓度:")
print({f'P{i}': float(c) for i, c in enumerate(target_p, start=1)})

D = 10
bounds = np.array([(0, 2.0)] * D)
max_iter = 10000 * D
pop_size = None

def run_optimizer(optimizer_name):
    tol = 1e-8
    if optimizer_name == "L-SHADE":
        result = LSHADE(objective, bounds, pop_size, max_iter, H=6, tol=tol)
    elif optimizer_name == "L-SHADE-cnEpsin":
        result = LSHADE_cnEpsin(objective, bounds, pop_size, max_iter, H=5, tol=tol)
    elif optimizer_name == "jSO":
        result = jSO(objective, bounds, pop_size, max_iter, H=5, tol=tol)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

    print(f"{optimizer_name} started.")
    _, _, history = result.optimize()
    print(f"{optimizer_name} completed. Final gap: {history[-1]}")
    return optimizer_name, np.array(history)

# 并行执行部分
if __name__ == "__main__":
    optimizers = ["L-SHADE", "L-SHADE-cnEpsin", "jSO"]

    num_workers = min(len(optimizers), cpu_count())

    with Pool(num_workers) as pool:
        results_list = pool.map(run_optimizer, optimizers)

    # 汇总结果
    results = dict(results_list)

    # 绘图
    plt.figure(figsize=(10, 6))
    for opt_name, gaps in results.items():
        plt.plot(gaps, label=opt_name, lw=2)

    plt.yscale('log')
    plt.xlabel('Function Evaluations')
    plt.ylabel('Fitness Gap (log scale)')
    plt.title('Optimization Gap Comparison on CEC2017 F1')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
