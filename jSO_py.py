import numpy as np
from scipy.stats import norm, cauchy

class jSO:
    def __init__(self, func, bounds, pop_size=None, max_evals=None, H=5, tol=None):
        """
        L-SHADE优化算法类
        后续改进的基础算法
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.N_init = int(round(25 * np.log(self.dim) * np.sqrt(self.dim))) if pop_size is None else pop_size
        self.N_min = 4
        self.max_evals = 10000 * self.dim if max_evals is None else max_evals
        self.H = H
        self.tol = tol
        self.num_evals = self.N_init
        self.gen = 0

        # 初始化历史记忆
        self.F_memory = np.ones(self.H) * 0.3
        self.CR_memory = np.ones(self.H) * 0.8
        self.hist_idx = 0

        # 初始化种群和存档
        self.N_current = self.N_init  # 当前种群大小
        self.archive_size = 2.6
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.N_init, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.archive = []
        self.iteration_log = []

    # 线性种群缩减策略
    def _linear_pop_size_reduction(self):

        return max(self.N_min, int(round(self.N_init - (self.N_init - self.N_min) * self.num_evals / self.max_evals)))

    # 边界处理
    def handle_boundary(self, individual, parent):
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]

        # 当个别超界时修正
        individual = np.where(individual < low, (low + parent) / 2, individual)
        individual = np.where(individual > high, (high + parent) / 2, individual)

        return individual

    def mutant(self, F, i, p_best_indices, max_gen):
        # current-to-pbest/1变异策略
        p_best_idx = np.random.choice(p_best_indices)
        p_best = self.pop[p_best_idx]

        # 选择r1
        r1 = self.pop[np.random.choice(np.delete(np.arange(self.N_current), i), 1, replace=False)].flatten()

        r2_idx = np.random.choice(np.arange(self.N_current + len(self.archive)), 1, replace=False)[0]  # 随机选择一个下标
        if r2_idx >= self.N_current:  # 如果下标大于种群大小，则从存档中选择
            r2_idx -= self.N_current
            r2 = self.archive[r2_idx]
        else:  # 否则从当前种群中选择
            r2 = self.pop[r2_idx].flatten()

        # 变异操作
        if self.gen < 0.2 * max_gen:
            Fw = F * 0.7
        elif self.gen < 0.4 * max_gen:
            Fw = F * 0.8
        else:
            Fw = F * 1.2

        # 变异操作
        mutant = self.pop[i] + Fw * (p_best - self.pop[i]) + F * (r1 - r2)
        mutant = self.handle_boundary(mutant, self.pop[i])

        return mutant

    def cross(self, mutant, i, CR):
        cross_chorm = self.pop[i].copy()
        j = np.random.randint(0, self.dim)  # 随机选择一个维度
        for k in range(self.dim):  # 对每个维度进行交叉
            if np.random.rand() < CR or k == j:  # 如果随机数小于交叉率或者维度为j
                cross_chorm[k] = mutant[k]  # 交叉
        return cross_chorm

    def optimize(self):
        i = 0
        max_gen = 0
        # 计算最大迭代次数
        while i < self.max_evals:
            max_gen += 1
            n = int(round(self.N_init - (self.N_init - self.N_min) * i / self.max_evals))
            i += n
        print(max_gen)

        """执行优化过程"""
        while self.num_evals < self.max_evals:
            S_F, S_CR, S_weights = [], [], []
            new_pop = []
            new_fitness = []

            # 记录当前最优值
            best_val = np.min(self.fitness)
            self.iteration_log.append(best_val)
            # 收敛检查
            if self.tol is not None and best_val <= self.tol:
                print(f"Converged at generation {self.gen + 1}: {best_val}")
                break
            elif self.num_evals > self.max_evals:
                print(f"Converged at generation {self.gen + 1}: {best_val}")
                break

            # current-to-pbest/1变异策略
            p = 0.125 + (0.25 - 0.125) * self.num_evals / self.max_evals
            p_best_size = max(2, int(self.N_current * p))
            p_best_indices = np.argsort(self.fitness)[:p_best_size]

            for i in range(self.N_current):
                # 从历史记忆随机选择索引
                r = np.random.randint(0, self.H)
                if r == self.H - 1:
                    self.CR_memory[r] = 0.9
                    self.F_memory[r] = 0.9

                # 生成F和CR
                if np.isnan(self.CR_memory[r]):
                    CR = 0
                else:
                    CR = np.clip(np.random.normal(self.CR_memory[r], 0.1), 0.0, 1.0)
                F = cauchy.rvs(loc=self.F_memory[r], scale=0.1)
                while F > 1 or F < 0:
                    if F < 0:
                        F = cauchy.rvs(loc=self.F_memory[r], scale=0.1)
                    else:
                        F = 1

                # jSO的CR阶段式调整（根据当前评估次数）
                if self.gen < 0.25 * max_gen:
                    CR = max(CR, 0.7)
                elif self.gen < 0.5 * max_gen:
                    CR = max(CR, 0.6)

                # jSO的F时变调整
                if self.gen < 0.6 * max_gen and F > 0.7:
                    F = 0.7

                # current-to-pbest/1变异策略
                mutant = self.mutant(F, i, p_best_indices, max_gen)

                # 交叉操作
                trial = self.cross(mutant, i, CR)
                trial_fitness = self.func(trial)

                # 贪心选择
                if trial_fitness < self.fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)
                    # 记录成功参数和适应度改进量
                    S_F.append(F)
                    S_CR.append(CR)
                    S_weights.append(np.abs(self.fitness[i] - trial_fitness))
                    # 更新存档
                    self.archive.append(self.pop[i].copy())
                    if len(self.archive) > self.archive_size * self.N_current:
                        # 如果存档超过种群大小，则随机删除一些
                        for o in range(int(len(self.archive) - self.archive_size * self.N_current)):
                            self.archive.pop(np.random.randint(0, len(self.archive)))
                else:
                    new_pop.append(self.pop[i])
                    new_fitness.append(self.fitness[i])
            self.num_evals += self.N_current
            self.gen += 1

            # --- LPSR关键步骤 ---
            # 更新种群大小
            new_N = self._linear_pop_size_reduction()
            # 选择适应度最好的个体保留
            combined_fitness = np.array(new_fitness)
            survivor_indices = np.argsort(combined_fitness)[:new_N]
            # 更新种群和适应度
            self.pop = np.array([new_pop[i] for i in survivor_indices])
            self.fitness = np.array([new_fitness[i] for i in survivor_indices])
            self.N_current = new_N

            # 更新历史记忆（加权Lehmer均值）
            if np.any(S_F):
                F_lehmer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
                self.F_memory[self.hist_idx] = (F_lehmer + self.F_memory[self.hist_idx]) / 2
            else:
                self.F_memory[self.hist_idx] = self.F_memory[self.hist_idx]

            # CR 部分
            if np.any(S_CR):
                if np.max(S_CR) != 0:
                    CR_lehmer = np.sum(np.array(S_CR) ** 2 * S_weights) / np.sum(np.array(S_CR) * S_weights)
                    self.CR_memory[self.hist_idx] = (CR_lehmer + self.CR_memory[self.hist_idx]) / 2
                else:
                    self.CR_memory[self.hist_idx] = 1

            # 移动历史指针
            self.hist_idx = (self.hist_idx + 1) % self.H

            if self.gen % 100 == 0 or self.gen > max_gen:
                print(f"Iteration {self.gen}, Pop Size: {self.N_current}, Best: {np.min(self.fitness)}, Num_Evals: {self.num_evals}")

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log