from collections import deque

import numpy as np
from scipy.stats import norm, cauchy

class mLSHADE_SPACMA:
    def __init__(self, func, bounds, pop_size=None, max_evals=None, H=6, tol=None):
        """
        L-SHADE优化算法类
        后续改进的基础算法
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.N_init = 18 * self.dim if pop_size is None else pop_size
        self.N_min = 4
        self.max_evals = 10000 * self.dim if max_evals is None else max_evals
        self.H = H
        self.tol = tol
        self.num_evals = self.N_init
        self.gen = 0
        self.rho = 0.11

        # 初始化历史记忆
        self.F_memory = np.ones(self.H) * 0.5
        self.CR_memory = np.ones(self.H) * 0.5
        self.hist_idx = 0

        # 混合算法参数
        self.FCP_memory = [0.5] * H  # First Class Probability (LSHADE分配概率)
        self.c = 0.8  # 学习率

        # 初始化种群和存档
        self.N_current = self.N_init  # 当前种群大小
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.N_init, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.archive = deque(maxlen=int(1.4 * self.N_current))  # FIFO存档（最大长度1.4*NP）
        self.iteration_log = []

        # CMA-ES参数初始化
        self.sigma = 0.5
        self.xmean = np.mean(self.pop, axis=0)
        self.mu = self.N_current // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights ** 2)

        # 协方差矩阵参数
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.dim + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs

        # 协方差矩阵
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        self.C = self.B @ np.diag(self.D ** 2) @ self.B.T
        self.invsqrtC = np.eye(self.dim)
        self.pc = np.zeros(self.dim)
        self.ps = np.zeros(self.dim)
        self.eigeneval = 0
        self.chiN = self.dim ** 0.5 * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

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

    def mutant(self, i, F, FCP):
        """生成变异个体（包含CMA-ES混合逻辑）"""
        if np.random.rand() < FCP:
            # 采用LSHADE变异策略
            p_i = 0.11
            p_best_size = max(2, int(self.N_current * p_i))
            sorted_indices = np.argsort(self.fitness)

            p_best_indices = sorted_indices[:p_best_size]
            p_best_idx = np.random.choice(p_best_indices)
            p_best = self.pop[p_best_idx]

            # 计算排名及概率
            ranks = 3 * (self.N_current - np.arange(1, self.N_current + 1)) + 1
            # 计算概率
            prs = np.zeros(self.N_current)
            prs[sorted_indices] = ranks
            prs /= prs.sum()

            # r1基于RSP概率
            candidates_r1 = np.setdiff1d(np.arange(self.N_current), [i, p_best_idx])
            prs_r1 = prs[candidates_r1]
            prs_r1 /= prs_r1.sum()
            r1_idx = np.random.choice(candidates_r1, p=prs_r1)
            r1 = self.pop[r1_idx]

            # r2从A∪P抽取
            if len(self.archive) > 0:
                combined = np.vstack([self.pop, np.array(self.archive)])
            else:
                combined = self.pop
            r2_idx = np.random.randint(0, combined.shape[0])
            r2 = combined[r2_idx]

            mutant = self.pop[i] + F * (p_best - self.pop[i]) + F * (r1 - r2)
            FCP_judge = 1
        else:
            # CMA-ES变异策略
            z = self.B @ (self.D * np.random.randn(self.dim))
            mutant = self.xmean + self.sigma * z
            FCP_judge = 2

        mutant = self.handle_boundary(mutant, self.pop[i])
        return mutant, FCP_judge
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
            S_FCP = []
            delta_alg1, delta_alg2 = [], []
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

            if self.gen < max_gen // 2:
                # 淘汰机制
                PE_m = int(np.ceil(self.rho * self.N_current))
                sorted_indices = np.argsort(self.fitness)
                eliminated_indices = sorted_indices[-PE_m:]

                # 生成新个体
                best1 = self.pop[sorted_indices[0]]
                best2 = self.pop[sorted_indices[1]]
                new_individuals = []

                for _ in range(PE_m):
                    # 生成新个体
                    new_ind = best1 + np.random.rand() * (best1 - best2)

                    # 边界处理 - 确保新个体在搜索空间内
                    for d in range(self.dim):
                        if new_ind[d] > self.bounds[:, 1][d]:
                            new_ind[d] = 2 * self.bounds[:, 1][d] - new_ind[d]
                        elif new_ind[d] < self.bounds[:, 0][d]:
                            new_ind[d] = 2 * self.bounds[:, 0][d] - new_ind[d]

                    new_individuals.append(new_ind)

            for i in range(self.N_current):
                # 参数生成阶段
                r = np.random.randint(0, self.H)

                CR = np.clip(np.random.normal(self.CR_memory[r], 0.1), 0, 1)

                # SPA机制：前半段固定F范围，后半段自适应
                if self.num_evals < self.max_evals / 2:
                    F = 0.5 + 0.1 * np.random.rand()
                else:
                    F = cauchy.rvs(loc=self.F_memory[r], scale=0.1)
                    while F > 1 or F < 0:
                        if F < 0:
                            F = cauchy.rvs(loc=self.F_memory[r], scale=0.1)
                        else:
                            F = 1

                FCP = self.FCP_memory[r]

                # 生成变异个体
                mutant, FCP_judge = self.mutant(i, F, FCP)
                trial = self.cross(mutant, self.pop[i], CR)

                trial_fitness = self.func(trial)

                # 贪心选择
                if trial_fitness < self.fitness[i]:
                    new_pop.append(trial)
                    new_fitness.append(trial_fitness)
                    # 记录成功参数和适应度改进量
                    S_F.append(F)
                    S_CR.append(CR)
                    S_FCP.append(FCP)
                    S_weights.append(np.abs(self.fitness[i] - trial_fitness))
                    # 更新存档
                    self.archive.append(self.pop[i].copy())
                    if FCP_judge == 1:
                        delta_alg1.append(self.fitness[i] - trial_fitness)
                    else:
                        delta_alg2.append(self.fitness[i] - trial_fitness)
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

            self.mu = max(1, self.N_current // 2)  # 防止mu为0
            self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
            self.weights /= np.sum(self.weights)  # 归一化权重
            self.mueff = 1 / np.sum(self.weights ** 2)  # 更新有效种群大小

            # 更新CMA-ES参数
            popold = np.copy(self.pop)  # CMA-ES部分用的旧种群

            # 按适应度排序
            popindex = np.argsort(self.fitness)
            if np.any(delta_alg2):
                xold = self.xmean.copy()
                self.xmean = np.dot(popold[popindex[:self.mu]].T, self.weights)

                # 演化路径ps更新
                y = (self.xmean - xold) / self.sigma
                self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * np.dot(
                    self.invsqrtC, y)

                # hsig判别
                ps_norm_sq = np.sum(self.ps ** 2)
                hsig_cond = (1 - (1 - self.cs) ** (2 * (self.num_evals + 1) / self.N_current))
                hsig = ps_norm_sq / (self.dim * hsig_cond) < (2 + 4 / (self.dim + 1))

                # 演化路径pc更新
                self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y

                # 协方差矩阵C更新
                artmp = (popold[popindex[:self.mu]] - xold) / self.sigma  # mu个差分向量
                self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (
                            np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (
                                2 - self.cc) * self.C) + self.cmu * np.dot((artmp.T * self.weights), artmp)

                # 步长sigma更新
                self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

                # C矩阵特征分解更新，O(d^2)复杂度
                if self.num_evals - self.eigeneval >= np.ceil(self.N_current / (self.c1 + self.cmu) / self.dim / 10):
                    self.eigeneval = (self.num_evals + 1)
                    self.C = np.triu(self.C) + np.triu(self.C, 1).T  # 保证对称性
                    if np.any(np.isnan(self.C)) or np.any(~np.isfinite(self.C)) or not np.isrealobj(self.C):
                        print("C matrix invalid, skipping CMA update")
                        continue  # 出问题跳过CMA-ES更新
                    D2, B = np.linalg.eigh(self.C)
                    if np.any(D2 < 0):
                        D2[D2 < 0] = 1e-10  # 数值安全
                    self.D = np.sqrt(D2)
                    self.B = B
                    self.invsqrtC = np.dot(self.B, np.dot(np.diag(self.D ** -1), self.B.T))

            # 更新历史记忆（加权Lehmer均值）
            if np.any(S_F):
                F_lehmer = np.sum(np.array(S_F) ** 2 * S_weights) / np.sum(np.array(S_F) * S_weights)
                self.F_memory[self.hist_idx] = F_lehmer

            # CR 部分
            if np.any(S_CR):
                if np.max(S_CR) == 0 or self.CR_memory[self.hist_idx] == np.nan:
                    self.CR_memory[self.hist_idx] = np.nan  # 置为 ⊥，表示未来采样的 CR 必为 0
                else:
                    CR_lehmer = np.sum(np.array(S_CR) ** 2 * S_weights) / np.sum(np.array(S_CR) * S_weights)
                    self.CR_memory[self.hist_idx] = CR_lehmer

            if np.any(S_FCP):
                # 更新混合概率
                total_improve = np.sum(delta_alg1) + np.sum(delta_alg2)
                ratio = np.sum(delta_alg1) / total_improve
                # 平滑更新公式
                self.FCP_memory[self.hist_idx] = np.clip(self.c * self.FCP_memory[self.hist_idx] + (1 - self.c) * ratio, 0.2, 0.8)

            # 移动历史指针
            self.hist_idx = (self.hist_idx + 1) % self.H

            if self.gen % 100 == 0 or self.gen > max_gen:
                print(f"Iteration {self.gen}, Pop Size: {self.N_current}, Best: {np.min(self.fitness)}, Num_Evals: {self.num_evals}")

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log