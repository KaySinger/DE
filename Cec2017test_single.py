from multiprocessing import Pool, cpu_count
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from LSHADE_py import LSHADE
from LSHADE_cnEpsin_py import LSHADE_cnEpsin
from jSO_py import jSO
from LSHADE_RSP_py import LSHADE_RSP
from APSM_jSO_py import APSM_jSO
from iLSHADE_RSP_py import iLSHADE_RSP
from ACD_DE_py import ACD_DE
from APDSDE_py import APDSDE
from mLSHADE_SPACMA_py import mLSHADE_SPACMA
import cec2017.functions as functions

def cec2017_f1(x):
    x = np.atleast_2d(x)
    val = functions.f4(x)
    return val[0]

D = 30
bounds = [(-100, 100)] * D
max_iter = 10000 * D
pop_size = None
optimal_value = 400

def run_optimizer(optimizer_name):
    objective = cec2017_f1
    tol = optimal_value + 1e-8
    if optimizer_name == "L-SHADE":
        result = LSHADE(objective, bounds, pop_size, max_iter, H=6, tol=None)
    elif optimizer_name == "L-SHADE-cnEpsin":
        result = LSHADE_cnEpsin(objective, bounds, pop_size, max_iter, H=5, tol=tol)
    elif optimizer_name == "jSO":
        result = jSO(objective, bounds, pop_size, max_iter, H=5, tol=tol)
    elif optimizer_name == "L-SHADE-RSP":
        result = LSHADE_RSP(objective, bounds, pop_size, max_iter, H=5, tol=tol)
    elif optimizer_name == "APSM-jSO":
        result = APSM_jSO(objective, bounds, pop_size, max_iter, H=6, tol=tol)
    elif optimizer_name == "iL-SHADE-RSP":
        result = iLSHADE_RSP(objective, bounds, pop_size, max_iter, H=5, tol=tol)
    elif optimizer_name == "ACD-DE":
        result = ACD_DE(objective, bounds, pop_size, max_iter, H=6, tol=None)
    elif optimizer_name == "APDSDE":
        result = APDSDE(objective, bounds, pop_size, max_iter, H=6, tol=None)
    elif optimizer_name == "mL-SHADE-SPACMA":
        result = mLSHADE_SPACMA(objective, bounds, pop_size, max_iter, H=5, tol=None)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

    print(f"{optimizer_name} started.")
    _, _, history = result.optimize()
    print(f"{optimizer_name} completed. Final gap: {history[-1] - optimal_value}")
    return optimizer_name, np.array(history) - optimal_value

# 并行执行部分
if __name__ == "__main__":
    # optimizers = ["L-SHADE", "L-SHADE-cnEpsin", "jSO", "L-SHADE-RSP", "APSM-jSO"]
    optimizers = ["mL-SHADE-SPACMA"]

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
