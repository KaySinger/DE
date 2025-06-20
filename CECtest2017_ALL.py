import numpy as np
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
import cec2017.functions as functions
import traceback
import time
import warnings

# 忽略特定警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 导入所有优化算法
from LSHADE_py import LSHADE
from jSO_py import jSO
from LSHADE_cnEpsin_py import LSHADE_cnEpsin
from LSHADE_RSP_py import LSHADE_RSP
from iLSHADE_RSP_py import iLSHADE_RSP
from ACD_DE_py import ACD_DE
from APDSDE_py import APDSDE
from APSM_jSO_py import APSM_jSO
# 设置全局参数
D = 30
bounds = [(-100, 100)] * D
max_iter = 10000 * D
pop_size = None

# CEC2017最优值（根据官方文档设置）
optimal_values = {
    1: 100, 3: 300, 4: 400, 5: 500, 6: 600, 7: 700, 8: 800, 9: 900, 10: 1000,
    11: 1100, 12: 1200, 13: 1300, 14: 1400, 15: 1500, 16: 1600, 17: 1700, 18: 1800,
    19: 1900, 20: 2000, 21: 2100, 22: 2200, 23: 2300, 24: 2400, 25: 2500, 26: 2600,
    27: 2700, 28: 2800, 29: 2900, 30: 3000
}

# 优化算法列表
optimizers = ["ACD-DE"]

# 测试函数列表 (15-30)
test_functions = list(range(1, 31))
test_functions.remove(2)

# 结果保存路径
base_dir = r"C:\Users\13119\Desktop\CEC测试"

# 运行次数
NUM_RUNS = 10


def create_cec_wrapper(func_index):
    """创建CEC2017函数的包装器，处理输入维度问题"""
    cec_func = getattr(functions, f"f{func_index}")

    def wrapper(x):
        # 确保输入是二维数组
        x = np.atleast_2d(x)
        return cec_func(x)[0]  # 返回单个值

    return wrapper


def get_optimizer(optimizer_name, objective, tol):
    """根据名称获取优化器实例"""
    if optimizer_name == "L-SHADE-cnEpsin":
        return LSHADE_cnEpsin(objective, bounds, pop_size, max_iter, H=5, tol=tol)
    elif optimizer_name == "iL-SHADE-RSP":
        return iLSHADE_RSP(objective, bounds, pop_size, max_iter, H=5, tol=tol)
    elif optimizer_name == "L-SHADE":
        return LSHADE(objective, bounds, pop_size, max_iter, H=6, tol=tol)
    elif optimizer_name == "APSM-jSO":
        return APSM_jSO(objective, bounds, pop_size, max_iter, H=5, tol=tol)
    elif optimizer_name == "APDSDE":
        return APDSDE(objective, bounds, pop_size, max_iter, H=6, tol=tol)
    elif optimizer_name == "ACD-DE":
        return ACD_DE(objective, bounds, pop_size, max_iter, H=6, tol=tol)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")


def run_single_optimization(args):
    """运行单次优化并返回结果"""
    optimizer_name, func_index, run_idx = args
    try:
        # 获取测试函数包装器
        objective = create_cec_wrapper(func_index)
        optimal = optimal_values[func_index]
        tol = optimal + 1e-8

        # 创建优化器
        optimizer = get_optimizer(optimizer_name, objective, tol)

        # 运行优化
        start_time = time.time()
        _, best_fitness, _ = optimizer.optimize()
        run_time = time.time() - start_time

        # 计算适应度间隔
        gap = best_fitness - optimal

        return {
            'optimizer': optimizer_name,
            'function': func_index,
            'run': run_idx,
            'best_fitness': best_fitness,
            'gap': gap,
            'run_time': run_time
        }
    except Exception as e:
        print(f"Error in {optimizer_name} on f{func_index} run {run_idx}: {str(e)}")
        traceback.print_exc()
        return None


def run_all_functions_for_optimizer(optimizer_name):
    """针对单个优化算法运行所有测试函数50次"""
    print(f"\n{'=' * 50}")
    print(f"开始测试优化器: {optimizer_name}")
    print(f"{'=' * 50}")

    # 创建算法专属目录
    algo_dir = os.path.join(base_dir, f"{optimizer_name}-cec2017")
    os.makedirs(algo_dir, exist_ok=True)

    # 遍历所有测试函数
    for func_idx in test_functions:
        print(f"\n处理函数 f{func_idx}...")

        # 创建函数专属目录
        func_dir = os.path.join(algo_dir, f"f{func_idx}")
        os.makedirs(func_dir, exist_ok=True)

        # 准备所有运行任务（50次运行）
        tasks = [(optimizer_name, func_idx, run_idx) for run_idx in range(NUM_RUNS)]

        # 存储结果
        all_results = []
        error_count = 0

        # 使用多进程并行执行50次运行
        print(f"启动 {NUM_RUNS} 次并行运行...")
        with Pool(processes=cpu_count()) as pool:
            for i, result in enumerate(pool.imap_unordered(run_single_optimization, tasks)):
                if result is not None:
                    all_results.append(result)
                else:
                    error_count += 1

                # 每完成10次运行打印一次进度
                if (i + 1) % 10 == 0:
                    print(f"已完成 {i + 1}/{NUM_RUNS} 次运行 (错误: {error_count})")

        # 检查是否有有效结果
        if not all_results:
            print(f"警告: {optimizer_name} 在函数 f{func_idx} 上没有获得有效结果")
            continue

        # 计算统计指标
        gaps = [r['gap'] for r in all_results]
        times = [r['run_time'] for r in all_results]

        stats = {
            'Algorithm': optimizer_name,
            'Function': func_idx,
            'Best': np.min(gaps),
            'Worst': np.max(gaps),
            'Median': np.median(gaps),
            'Mean': np.mean(gaps),
            'Std': np.std(gaps),
            'Avg_Time': np.mean(times),
            'Min_Time': np.min(times),
            'Max_Time': np.max(times),
            'Success_Rate': np.sum(np.array(gaps) <= 1e-8) / len(all_results) * 100,
            'Num_Runs': len(all_results),
            'Error_Runs': error_count
        }

        # 保存详细结果
        results_df = pd.DataFrame(all_results)
        detailed_file = os.path.join(func_dir, f"f{func_idx}_detailed_results.csv")
        results_df.to_csv(detailed_file, index=False)
        print(f"详细结果已保存至: {detailed_file}")

        # 保存统计结果
        stats_df = pd.DataFrame([stats])
        stats_file = os.path.join(func_dir, f"f{func_idx}_statistics.csv")
        stats_df.to_csv(stats_file, index=False)
        print(f"统计结果已保存至: {stats_file}")

        # 打印统计摘要
        print(f"\n{optimizer_name} 在 f{func_idx} 上的统计结果 (基于 {len(all_results)} 次运行):")
        print(f"最佳差距: {stats['Best']:.4e}")
        print(f"最差差距: {stats['Worst']:.4e}")
        print(f"平均差距: {stats['Mean']:.4e} ± {stats['Std']:.4e}")
        print(f"中值差距: {stats['Median']:.4e}")
        print(f"平均时间: {stats['Avg_Time']:.2f}秒 (最小: {stats['Min_Time']:.2f}秒, 最大: {stats['Max_Time']:.2f}秒)")
        print(f"成功率: {stats['Success_Rate']:.1f}%")
        print(f"错误运行次数: {error_count}")

    print(f"\n{'=' * 50}")
    print(f"完成优化器 {optimizer_name} 的所有测试")
    print(f"{'=' * 50}")


def main():
    """主测试函数"""
    # 创建基础目录
    os.makedirs(base_dir, exist_ok=True)

    # 遍历所有优化算法
    for optimizer in optimizers:
        run_all_functions_for_optimizer(optimizer)


if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"\n所有测试完成！总耗时: {total_time / 3600:.2f} 小时")