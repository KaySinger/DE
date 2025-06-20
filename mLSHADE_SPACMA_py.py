import numpy as np
from scipy.stats import norm, cauchy
import collections


class mLSHADE_SPACMA:
    def __init__(self, func, bounds, pop_size=None, max_evals=None, H=5, tol=None):
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.N_init = 18 * self.dim if pop_size is None else pop_size
        self.N_min = 4
        self.max_evals = 10000 * self.dim if max_evals is None else max_evals
        self.H = H
        self.tol = tol
        self.num_evals = 0
        self.gen = 1
        self.rho = 0.11  # 淘汰比例
        self.k = 3  # RSP控制参数

        # 历史记忆初始化
        self.F_memory = np.ones(self.H) * 0.5
        self.CR_memory = np.ones(self.H) * 0.5
        self.hist_idx = 0

        # 混合算法参数
        self.FCP_memory = np.ones(self.H) * 0.5  # LSHADE分配概率
        self.c = 0.8  # 学习率
        self.L_rate = 0.8  # 混合参数学习率

        # 初始化种群
        self.N_current = self.N_init
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.N_init, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.num_evals += self.N_init
        self.iteration_log = []

        # 存档初始化 (修改为列表存储)
        self.archive = []  # 格式: (solution, fitness)
        self.archive_max_size = int(1.4 * self.N_current)

        # 找出初始最优解
        best_idx = np.argmin(self.fitness)
        self.best_fitness = self.fitness[best_idx]
        self.best_solution = self.pop[best_idx].copy()
        self.iteration_log.append(self.best_fitness)

        # CMA-ES参数初始化 (修改为加权平均)
        sorted_indices = np.argsort(self.fitness)
        self.mu = self.N_current // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.xmean = np.dot(self.weights, self.pop[sorted_indices[:self.mu]])

        # 协方差矩阵参数
        self.sigma = 0.5
        self.mueff = 1 / np.sum(self.weights ** 2)
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.C = self.B @ np.diag(self.D ** 2) @ self.B.T
        self.invsqrtC = np.eye(self.dim)
        self.eigeneval = 0
        self.chiN = self.dim ** 0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

        # 计算最大迭代次数
        self.max_gen = 0
        i = 0
        while i < self.max_evals:
            self.max_gen += 1
            n = int(round(self.N_init - (self.N_init - self.N_min) * i / self.max_evals))
            i += n

    def _linear_pop_size_reduction(self):
        return max(self.N_min, int(round(
            self.N_init - (self.N_init - self.N_min) * self.num_evals / self.max_evals
        )))

    def bound_constraint(self, vi, parent):
        """边界处理，使用反射策略 (与作者Matlab代码一致)"""
        for d in range(self.dim):
            low, high = self.bounds[d]
            if vi[d] < low:
                vi[d] = (parent[d] + low) / 2
            elif vi[d] > high:
                vi[d] = (parent[d] + high) / 2
        return vi

    def generate_mutant(self, i, F, FCP):
        """生成变异个体 (整合RSP和CMA-ES)"""
        if np.random.rand() < FCP:  # LSHADE变异
            # 计算排名概率 (修正RSP实现)
            sorted_indices = np.argsort(self.fitness)
            ranks = self.k * (self.N_current - np.arange(self.N_current)) + 1
            prs = ranks / np.sum(ranks)

            # 选择pbest (前p%)
            p_i = 0.11
            p_best_size = max(2, int(self.N_current * p_i))
            p_best_idx = sorted_indices[np.random.randint(0, p_best_size)]
            p_best = self.pop[p_best_idx]

            # 基于排名概率选择r1
            r1_idx = np.random.choice(self.N_current, p=prs)
            r1 = self.pop[sorted_indices[r1_idx]]

            # 从种群或存档中选择r2
            if len(self.archive) > 0:
                archive_solutions = np.array([item[0] for item in self.archive])
                combined = np.vstack([self.pop, archive_solutions])
            else:
                combined = self.pop
            r2_idx = np.random.randint(0, len(combined))
            r2 = combined[r2_idx]

            mutant = self.pop[i] + F * (p_best - self.pop[i]) + F * (r1 - r2)
            strategy_type = 1  # LSHADE
        else:  # CMA-ES变异
            z = self.B @ (self.D * np.random.randn(self.dim))
            mutant = self.xmean + self.sigma * z
            strategy_type = 2  # CMA-ES

        mutant = self.bound_constraint(mutant, self.pop[i])
        return mutant, strategy_type

    def crossover(self, mutant, target, CR):
        """二项交叉"""
        trial = np.copy(target)
        j_rand = np.random.randint(0, self.dim)
        for j in range(self.dim):
            if np.random.rand() < CR or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def update_archive(self, solution, fitness):
        """更新存档 (精英保留策略)"""
        self.archive.append((solution.copy(), fitness))

        # 如果存档超过大小，删除最差个体
        if len(self.archive) > self.archive_max_size:
            # 按适应度排序 (最小最优)
            self.archive.sort(key=lambda x: x[1])
            # 保留最好的个体
            self.archive = self.archive[:self.archive_max_size]

    def optimize(self):
        print(self.max_gen)
        """执行优化过程"""
        while self.num_evals < self.max_evals:
            # 记录当前最优
            self.iteration_log.append(self.best_fitness)

            # 收敛检查
            if self.tol is not None and self.best_fitness <= self.tol:
                break
            if self.num_evals >= self.max_evals:
                break

            # 精确淘汰与生成机制 (仅在前半段)
            if self.num_evals < self.max_evals / 2:
                PE_m = int(np.ceil(self.rho * self.N_current))
                sorted_indices = np.argsort(self.fitness)

                # 生成新个体
                best1 = self.pop[sorted_indices[0]]
                best2 = self.pop[sorted_indices[1]]
                new_individuals = []

                for _ in range(PE_m):
                    new_ind = best1 + np.random.rand() * (best1 - best2)
                    new_ind = self.bound_constraint(new_ind, best1)
                    new_individuals.append(new_ind)

                # 替换最差个体
                self.pop[sorted_indices[-PE_m:]] = new_individuals
                self.fitness[-PE_m:] = np.apply_along_axis(self.func, 1, new_individuals)
                self.num_evals += PE_m

                # 更新最优解
                current_best_idx = np.argmin(self.fitness)
                if self.fitness[current_best_idx] < self.best_fitness:
                    self.best_fitness = self.fitness[current_best_idx]
                    self.best_solution = self.pop[current_best_idx].copy()

            # 初始化成功参数记录
            S_F, S_CR, S_weights = [], [], []
            delta_alg1, delta_alg2 = [], []
            new_pop = []
            new_fitness = []

            # 为每个个体生成参数
            mem_rand_index = np.random.randint(0, self.H, self.N_current)
            mu_sf = self.F_memory[mem_rand_index]
            mu_cr = self.CR_memory[mem_rand_index]
            FCP_rand = self.FCP_memory[mem_rand_index]

            # 生成CR和F
            cr = np.clip(np.random.normal(mu_cr, 0.1), 0, 1)
            term_pos = mu_cr == -1
            cr[term_pos] = 0

            if self.num_evals < self.max_evals / 2:
                sf = 0.5 + 0.1 * np.random.rand(self.N_current)
            else:
                sf = mu_sf + 0.1 * np.tan(np.pi * (np.random.rand(self.N_current) - 0.5))
                sf = np.clip(sf, 0, 1)

            # 主循环处理每个个体
            for i in range(self.N_current):
                # 生成变异个体
                mutant, strategy_type = self.generate_mutant(i, sf[i], FCP_rand[i])

                # 交叉
                trial = self.crossover(mutant, self.pop[i], cr[i])

                # 评估
                trial_fitness = self.func(trial)

                # 更新全局最优
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial.copy()

                # 贪心选择
                if trial_fitness < self.fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)

                    # 记录成功参数
                    S_F.append(sf[i])
                    S_CR.append(cr[i])
                    S_weights.append(self.fitness[i] - trial_fitness)  # 改进量

                    # 更新存档
                    self.update_archive(self.pop[i], self.fitness[i])

                    # 记录策略改进量
                    if strategy_type == 1:
                        delta_alg1.append(self.fitness[i] - trial_fitness)
                    else:
                        delta_alg2.append(self.fitness[i] - trial_fitness)
                else:
                    new_pop.append(self.pop[i])
                    new_fitness.append(self.fitness[i])

            # 更新种群
            self.num_evals += self.N_current
            self.pop = np.array(new_pop)
            self.fitness = np.array(new_fitness)

            # 更新历史记忆
            if S_F:
                total_improvement = sum(S_weights)
                if total_improvement > 0:
                    # 归一化权重
                    norm_weights = [w / total_improvement for w in S_weights]

                    # 更新F记忆 (加权Lehmer均值)
                    F_lehmer = sum(f ** 2 * w for f, w in zip(S_F, norm_weights)) / sum(
                        f * w for f, w in zip(S_F, norm_weights))
                    self.F_memory[self.hist_idx] = F_lehmer

                    # 更新CR记忆
                    if max(S_CR) == 0 or self.CR_memory[self.hist_idx] == -1:
                        self.CR_memory[self.hist_idx] = -1
                    else:
                        CR_lehmer = sum(c ** 2 * w for c, w in zip(S_CR, norm_weights)) / sum(
                            c * w for c, w in zip(S_CR, norm_weights))
                        self.CR_memory[self.hist_idx] = CR_lehmer

                    # 更新FCP记忆
                    if delta_alg1 and delta_alg2:
                        ratio = sum(delta_alg1) / (sum(delta_alg1) + sum(delta_alg2))
                        self.FCP_memory[self.hist_idx] = self.L_rate * self.FCP_memory[self.hist_idx] + (
                                    1 - self.L_rate) * ratio
                        self.FCP_memory[self.hist_idx] = np.clip(self.FCP_memory[self.hist_idx], 0.2, 0.8)

                # 移动历史指针
                self.hist_idx = (self.hist_idx + 1) % self.H

            # 种群缩减
            new_N = self._linear_pop_size_reduction()
            if self.N_current > new_N:
                reduction_num = self.N_current - new_N
                sorted_indices = np.argsort(self.fitness)

                # 删除最差个体
                self.pop = np.delete(self.pop, sorted_indices[-reduction_num:], axis=0)
                self.fitness = np.delete(self.fitness, sorted_indices[-reduction_num:])
                self.N_current = new_N

                # 更新存档大小
                if len(self.archive) > self.archive_max_size:
                    self.archive.sort(key=lambda x: x[1])
                    self.archive = self.archive[:self.archive_max_size]

                # 更新CMA-ES参数
                self.mu = max(1, self.N_current // 2)
                self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
                self.weights /= np.sum(self.weights)
                self.mueff = 1 / np.sum(self.weights ** 2)

            # CMA-ES参数更新
            if self.num_evals > self.max_evals / 2:  # 仅在后半段更新
                sorted_indices = np.argsort(self.fitness)
                xold = self.xmean.copy()

                # 使用精英加权平均
                self.xmean = np.dot(self.weights, self.pop[sorted_indices[:self.mu]])

                # 演化路径更新
                y = (self.xmean - xold) / self.sigma
                self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * self.invsqrtC @ y

                # hsig判别
                ps_norm_sq = np.sum(self.ps ** 2)
                hsig_cond = (1 - (1 - self.cs) ** (2 * self.num_evals / self.N_current))
                hsig = ps_norm_sq / (self.dim * hsig_cond) < (2 + 4 / (self.dim + 1))

                # 演化路径pc更新
                self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y

                # 协方差矩阵更新
                artmp = (self.pop[sorted_indices[:self.mu]] - xold) / self.sigma
                self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (
                        np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C
                ) + self.cmu * artmp.T @ np.diag(self.weights) @ artmp

                # 步长更新
                self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

                # 定期更新特征分解
                if self.num_evals - self.eigeneval > self.N_current / (self.c1 + self.cmu) / self.dim / 10:
                    self.eigeneval = self.num_evals
                    self.C = (self.C + self.C.T) / 2  # 确保对称
                    D2, B = np.linalg.eigh(self.C)
                    D2 = np.maximum(D2, 1e-10)  # 防止负值
                    self.D = np.sqrt(D2)
                    self.B = B
                    self.invsqrtC = B @ np.diag(1 / self.D) @ B.T

            # 进度输出
            # if self.gen % 100 == 0:
            print(f"Gen {self.gen}, Pop: {self.N_current}, Best: {self.best_fitness:.6e}, Evals: {self.num_evals}/{self.max_evals}")

            self.gen += 1

        return self.best_solution, self.best_fitness, self.iteration_log