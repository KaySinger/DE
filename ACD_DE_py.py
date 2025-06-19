import numpy as np
from scipy.stats import norm, cauchy
from sklearn.cluster import KMeans


class ACD_DE:
    def __init__(self, func, bounds, pop_size=None, max_evals=None, H=6, tol=None):
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
        self.F_memory = np.ones(self.H) * 0.5
        self.CR_memory = np.ones(self.H) * 0.9
        self.hist_idx = 0
        self.p = 0.15
        self.pa = 0.2

        # 初始化种群和存档
        self.N_current = self.N_init  # 当前种群大小
        self.archive_size = 1.3
        self.pop = np.random.uniform(
            low=self.bounds[:, 0],
            high=self.bounds[:, 1],
            size=(self.N_init, self.dim)
        )
        self.fitness = np.apply_along_axis(self.func, 1, self.pop)
        self.archive = []
        self.iteration_log = []

        self.stagnation_counter = np.zeros(self.N_init, dtype=int)

        # 计算初始多样性指标
        center = np.mean(self.pop, axis=0)
        self.DI_init = np.sqrt(np.sum((self.pop - center) ** 2))

        # 初始聚类标签
        self.cluster_labels = np.zeros(self.N_init, dtype=int)

    # 混合种群缩减策略
    def _hybrid_pop_size_reduction(self, max_gen):
        cutoff = int(0.66 * max_gen)
        P_mid = int(0.33 * self.N_init)
        if self.gen < cutoff:
            N = int(round(self.N_init - (self.N_init - P_mid) * (self.gen / cutoff) ** 2))
        else:
            N = int(round(P_mid - (P_mid - self.N_min) * ((self.gen - cutoff) / (max_gen - cutoff))))

        return max(self.N_min, N)

    # 边界处理
    def handle_boundary(self, individual, parent):
        low = self.bounds[:, 0]
        high = self.bounds[:, 1]

        # 当个别超界时修正
        individual = np.where(individual < low, (low + parent) / 2, individual)
        individual = np.where(individual > high, (high + parent) / 2, individual)

        return individual

    def _cluster_population(self, k):
        kmeans = KMeans(n_clusters=k, n_init='auto')
        labels = kmeans.fit_predict(self.pop)
        return labels

    def _update_clusters(self):
        if self.gen == 0:
            self.cluster_labels = self._cluster_population(4)
        elif self.gen == int(0.66 * self.max_evals):
            self.cluster_labels = self._cluster_population(2)

    def _build_Cpn_set(self):
        """构建精英集合Cpn（论文4.1节）"""
        best_idx = np.argmin(self.fitness)
        best_cluster = self.cluster_labels[best_idx]

        # 获取最优簇内个体
        cluster_indices = np.where(self.cluster_labels == best_cluster)[0]
        cluster_size = len(cluster_indices)

        # 计算Cpn大小（公式8）
        n_cpn = max(2, int(self.p * self.N_current))

        # 选择簇内前10%的个体
        n_select = max(1, int(0.1 * cluster_size))
        sorted_indices = np.argsort(self.fitness[cluster_indices])[:n_select]
        Cpn_indices = cluster_indices[sorted_indices]

        # 如果数量不足，从全局补充适应度高的个体
        if len(Cpn_indices) < n_cpn:
            remaining = n_cpn - len(Cpn_indices)
            all_indices = np.setdiff1d(np.arange(self.N_current), Cpn_indices)
            sorted_global = np.argsort(self.fitness[all_indices])[:remaining]
            Cpn_indices = np.concatenate([Cpn_indices, all_indices[sorted_global]])

        return Cpn_indices

    def _update_archive(self):
        unique_clusters = np.unique(self.cluster_labels)
        for c in unique_clusters:
            indices = np.where(self.cluster_labels == c)[0]
            n_select = max(1, int(len(indices) * self.pa))
            selected = np.random.choice(indices, n_select, replace=False)
            for s in selected:
                self.archive.append(self.pop[s].copy())
        max_archive = int(self.archive_size * self.N_current)
        if len(self.archive) > max_archive:
            excess = len(self.archive) - max_archive
            remove_indices = np.random.choice(len(self.archive), excess, replace=False)
            self.archive = [a for i, a in enumerate(self.archive) if i not in remove_indices]

    def cross(self, mutant, i, CR):
        cross_chorm = self.pop[i].copy()
        j = np.random.randint(0, self.dim)  # 随机选择一个维度
        for k in range(self.dim):  # 对每个维度进行交叉
            if np.random.rand() < CR or k == j:  # 如果随机数小于交叉率或者维度为j
                cross_chorm[k] = mutant[k]  # 交叉
        return cross_chorm

    def _calculate_VIX(self, parent, offspring):
        """计算波动指数VIX（论文公式13）"""
        delta = offspring - parent
        if self.dim == 1:
            # 一维问题特殊处理
            return np.abs(delta[0])
        mean_delta = np.mean(delta)
        variance = np.sum((delta - mean_delta) ** 2) / (self.dim - 1)
        return np.sqrt(variance)

    def _update_parameters(self, S_F, S_CR, S_VIX):
        # 计算权重（VIX归一化）
        if not S_F or not S_CR or not S_VIX:
            return  # 无成功样本时不更新

        # 计算权重（公式14）
        total_VIX = sum(S_VIX)
        weights = [vix / total_VIX for vix in S_VIX]

        # 计算F的加权Lehmer均值（公式15）
        numerator_F = sum(w * f ** 2 for w, f in zip(weights, S_F))
        denominator_F = sum(w * f for w, f in zip(weights, S_F))
        F_lehmer = numerator_F / denominator_F if denominator_F != 0 else 0.5

        # 计算CR的加权Lehmer均值
        numerator_CR = sum(w * cr ** 2 for w, cr in zip(weights, S_CR))
        denominator_CR = sum(w * cr for w, cr in zip(weights, S_CR))
        CR_lehmer = numerator_CR / denominator_CR if denominator_CR != 0 else 0.9

        # 更新历史记忆
        self.F_memory[self.hist_idx] = F_lehmer
        self.CR_memory[self.hist_idx] = CR_lehmer

        # 移动历史指针
        self.hist_idx = (self.hist_idx + 1) % self.H

    def optimize(self):
        max_gen = 1130
        """执行优化过程"""
        while self.num_evals < self.max_evals:
            S_F, S_CR, S_VIX = [], [], []
            new_pop = []
            new_fitness = []
            F_list, p_best_list = [], []

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

            # 在关键代数更新聚类
            if self.gen in [0, int(0.66 * self.max_evals)]:
                self._update_clusters()

            # 构建Cpn集合（每个个体独立）
            Cpn_indices = self._build_Cpn_set()

            # 更新存档
            self._update_archive()

            self.p = 0.15 + (self.num_evals / self.max_evals) * 0.3
            self.pa = 0.2 - (self.num_evals / self.max_evals) * 0.05

            for i in range(self.N_current):
                # 从历史记忆随机选择索引
                r = np.random.randint(0, self.H)

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

                # current-to-Cpn/1变异策略
                F_list.append(F)

                # current-to-pbest/1变异策略
                p_best = self.pop[np.random.choice(Cpn_indices)]
                p_best_list.append(p_best)

                available_indices = np.delete(np.arange(self.N_current), i)
                r1 = self.pop[np.random.choice(available_indices)]

                r2 = self.archive[np.random.randint(len(self.archive))]

                mutant = self.pop[i] + F * (p_best - self.pop[i]) + F * (r1 - r2)
                mutant = self.handle_boundary(mutant, self.pop[i])

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
                    S_VIX.append(self._calculate_VIX(self.pop[i], trial))
                else:
                    new_pop.append(self.pop[i])
                    new_fitness.append(self.fitness[i])
            self.num_evals += self.N_current
            self.gen += 1

            # STAR机制：检测多样性，个体停滞后扰动或重启
            center = np.mean(new_pop, axis=0)
            DI_current = np.sqrt(np.sum((np.array(new_pop) - center) ** 2))
            RDI = DI_current / self.DI_init

            best_idx_new = np.argmin(new_fitness)
            best_solution = p_best_list[np.random.choice(len(p_best_list))]

            if RDI < 0.01:
                for i in range(self.N_current):
                    # 扰动阶段（停滞代数≥D）
                    if self.stagnation_counter[i] >= self.dim and i != best_idx_new:
                        # 选择维度（标准差最大的30%）
                        std_per_dim = np.std(new_pop, axis=0)
                        dim_indices = np.argsort(std_per_dim)[:int(0.3 * self.dim)]

                        # 随机选择个体
                        r1 = np.random.choice(np.delete(np.arange(self.N_current), i))
                        r2_archive = self.archive[np.random.randint(len(self.archive))] if self.archive else new_pop[r1]

                        # 公式11：扰动操作
                        new_pop[i][dim_indices] += F_list[i] * (best_solution[dim_indices] - new_pop[i][dim_indices]) + F_list[i] * (new_pop[r1][dim_indices] - r2_archive[dim_indices])

                        # 边界处理
                        new_pop[i] = self.handle_boundary(new_pop[i], self.pop[i])
                        new_fitness[i] = self.func(new_pop[i])

                    # 重启阶段（停滞代数≥D+15）
                    elif self.stagnation_counter[i] >= self.dim + 15 and i != best_idx_new:
                        # 随机选择个体
                        r1 = np.random.choice(np.delete(np.arange(self.N_current), i))
                        r2_archive = self.archive[np.random.randint(len(self.archive))] if self.archive else new_pop[r1]

                        # 公式12：重启操作
                        new_pop[i] = best_solution + F_list[i] * (new_pop[r1] - r2_archive)

                        # 边界处理
                        new_pop[i] = self.handle_boundary(new_pop[i], self.pop[i])
                        new_fitness[i] = self.func(new_pop[i])
                        self.stagnation_counter[i] = 0  # 重置计数器

            # --- LPSR关键步骤 ---
            # 更新种群大小
            new_N = self._hybrid_pop_size_reduction(max_gen)
            # 选择适应度最好的个体保留
            combined_fitness = np.array(new_fitness)
            survivor_indices = np.argsort(combined_fitness)[:new_N]
            # 更新种群和适应度
            self.pop = np.array([new_pop[i] for i in survivor_indices])
            self.fitness = np.array([new_fitness[i] for i in survivor_indices])
            self.N_current = new_N
            self.stagnation_counter = self.stagnation_counter[survivor_indices]
            self.cluster_labels = self.cluster_labels[survivor_indices]

            # 更新历史记忆（加权Lehmer均值）
            self._update_parameters(S_F, S_CR, S_VIX)

            # 移动历史指针
            self.hist_idx = (self.hist_idx + 1) % self.H

            if self.gen % 100 == 0 or self.gen > max_gen:
                print(f"Iteration {self.gen}, Pop Size: {self.N_current}, Best: {np.min(self.fitness)}, Num_Evals: {self.num_evals}")

        best_idx = np.argmin(self.fitness)
        return self.pop[best_idx], self.fitness[best_idx], self.iteration_log