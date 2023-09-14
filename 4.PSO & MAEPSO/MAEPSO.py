import numpy as np
import Benchmark
import matplotlib.pyplot as plt
from Benchmark import Benchmark

class MAEPSO:
    """
    MAEPSO算法
    author:holoxiong@stu.xmu.edu.cn
    date:2022.11.19
    """
    def __init__(self, fitness_fun) -> None:
        """
        MAEPSO算法的参数设置
        Tablet:(-100, 100) Quadric:(-100, 100) Rosenbrock:(-50, 50),
        Griewank:(-300, 300) Rastrigin:(-5.12, 5.12) Schaffer F7:(-100, 100)
        """
        self.popu_size = 20  # 种群大小
        self.dim = 30        # 函数维度
        self.M = 5           # 尺度算子
        self.W = 100         # 搜索空间宽度(见注释)
        self.c1 = 1.4        # 个体学习因子
        self.c2 = 1.4        # 全局学习因子
        self.w_range = (0.95, 0.4) # 惯性因子范围
        self.w = self.w_range[0]   # 初始惯性因子
        self.fitness_fun = fitness_fun  # 适应度函数
        self.P =  (int)(self.popu_size / self.M)  # 子群粒子个数

    # 初始化种群
    def init_population(self):
        self.X = np.random.uniform(-self.W, self.W, (self.popu_size, self.dim))  # 范围内(-100, 100)随机采样，得到粒子位置(大小（20，30）)
        self.pbest = np.copy(self.X)    # 个体最优位置
        min_index = np.argmin(self.get_fitness(self.pbest))
        self.gbest = self.pbest[min_index]  # 全局最优位置
        self.V = np.random.uniform(-1, 1, (self.popu_size, self.dim))  # 范围内随机采样，得到粒子初始速度
        self.T = np.ones(self.dim) * 0.5  # 速度阈值
        self.G = np.zeros(self.dim)       # 逃逸次数
        self.sigma = np.ones(self.M) * 2 * self.W   # 初始化高斯变异算子方差（优化变量取值范围）

    # 计算粒子群的适应度
    def get_fitness(self, X):
        fit = []
        for x in X:
            fit.append(self.fitness_fun(x))
        return np.array(fit)
    
    # 计算1个子群的平均适应度（按论文中的公式）
    def get_subgroup_fitness(self, subgroup):
        subgroup_fitness = self.get_fitness(subgroup)
        subgroup_fitness = np.sum(subgroup_fitness) / self.P  # 子群平均适应度
        return subgroup_fitness
    
    # 更新速度
    def update_velocity(self, i, j):
        # 根据公式更新速度
        new_v = self.w*self.V[i][j] + self.c1*np.random.rand()*(self.pbest[i][j]-self.X[i][j]) + \
            self.c2*np.random.rand()*(self.gbest[j]-self.X[i][j])
        self.V[i][j] = new_v

    # 更新最优位置
    def update_best_pos(self, i):
        # 个体
        if self.fitness_fun(self.X[i]) < self.fitness_fun(self.pbest[i]):
                    self.pbest[i] = self.X[i]
        # 全局
        if self.fitness_fun(self.pbest[i]) < self.fitness_fun(self.gbest):
            self.gbest = self.pbest[i]

    # 速度更新 逃逸 位置更新
    def update_velocity_escape_pos(self):
        for i in range(self.popu_size):
            for j in range(self.dim):
                self.update_velocity(i, j)

                # 按照论文中的变异操作进行逃逸
                if np.abs(self.V[i][j]) < self.T[j]:
                    # 多尺度变异
                    randn = np.random.randn(self.M)  # 具有正态分布特征的随机样本 长度：5
                    src_x = self.X[i][j]
                    min_fit = float("inf")
                    min_index = 0
                    for m in range(self.M):
                        self.X[i][j] = (src_x + randn[m]*self.sigma[m])
                        if self.fitness_fun(self.X[i]) < min_fit:
                            min_fit = self.fitness_fun(self.X[i])
                            min_index = m

                    v_max = self.W - abs(src_x)  # 1当前粒子可以逃逸的最大速度（即要在范围内）
                    rand = np.random.uniform(-1, 1)
                    self.X[i][j] = src_x + rand * v_max
                    if(min_fit < self.fitness_fun(self.X[i])):
                        self.V[i][j] = randn[min_index] * self.sigma[min_index]
                    else:
                        self.V[i][j] =  rand * v_max
                    # 位置还原
                    self.X[i][j] = src_x
                    # 增加逃逸次数
                    self.G[j] += 1

                # 更新粒子位置
                self.X[i][j] += self.V[i][j]
                # 更新最优位置    
                self.update_best_pos(i)
                       
    # 更新多尺度算子
    def update_M(self):
        sub_fit = []
        # 划分子群
        for m in range(self.M):
            subgroup = self.X[m*self.P : (m+1)*self.P]           
            sub_fit.append(self.get_subgroup_fitness(subgroup))  # 按照公式计算子群适应度
        sub_fit = np.array(sub_fit)  # 对Tablet：4个子群的平均适应度
        min_fit = np.min(sub_fit)
        max_fit = np.max(sub_fit)
        
        # 更新标准差
        for m in range(self.M):
            self.sigma[m] = self.sigma[m] * np.exp((self.M*sub_fit[m]-np.sum(sub_fit)) / (max_fit - min_fit + pow(10, -10))) #pow(10, -10)
            # 变异算子标准差规定
            self.sigma[m] %= self.W/4
    
    # 更新阈值
    def update_T(self):
        for i in range(self.dim):
            if self.G[i] > 5:               # k1 = 5
                self.G[i] = 0
                self.T[i] = self.T[i] / 10  # k2 = 10

    # 更新惯性因子
    def update_w(self, gene, generation):
        self.w = self.w_range[0] - (self.w_range[0]-self.w_range[1]) * gene / generation

    # 画图
    def draw_log_fitness(self, fun, fitness, epoch):
        plt.figure()
        fitness = np.array(fitness)
        plt.plot(np.log(fitness))
        plt.xlabel('generation')
        plt.ylabel('Log(fitness)')
        plt.title(fun)
        # plt.show()
        plt.savefig('./results/' + fun + str(epoch) + '.png')

    # 粒子群进化
    def particle_evolution(self, fun):
        for epoch in range(1):
            # 种群初始化
            self.init_population()
            fitness = []
            for gene in range(6000):
                # 更新速度 速度逃逸 位置更新
                self.update_velocity_escape_pos()
                # 更新多尺度变异算子
                self.update_M()
                # 更新速度阈值
                self.update_T()
                # 更新惯性因子
                self.update_w(gene, 6000)

                gbest_fit = self.fitness_fun(self.gbest)
                fitness.append(gbest_fit)
                if gbest_fit == 0:
                    print("finished")
                    print("gene:{} min_fitness:0".format(gene))
                    break
                if gene % 200 == 0:
                    print("gene:{} min_fitness:{}".format(gene, gbest_fit))
            print("epoch:{} min_fitness:{}".format(epoch, gbest_fit))
            print('\n')
            # self.draw_log_fitness(fun, fitness, epoch)
        return fitness  
    

