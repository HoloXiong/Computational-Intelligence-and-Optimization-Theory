import numpy as np
import matplotlib.pyplot as plt
from Benchmark import Benchmark

class PSO:
    """
    传统PSO算法
    author:holoxiong@stu.xmu.edu.cn
    date:2022.11.19
    """
    def __init__(self, fitness_fun) -> None:
        """
        PSO算法的参数设置
        """
        self.popu_size = 20  # 种群大小
        self.dim = 30        # 函数维度
        self.W = 100         # 搜索空间宽度(见注释)
        self.c1 = 1.4        # 个体学习因子
        self.c2 = 1.4        # 全局学习因子
        self.w_range = (0.95, 0.4) # 惯性因子范围
        self.w = self.w_range[0]   # 初始惯性因子
        self.fitness_fun = fitness_fun  # 适应度函数
    
    # 初始化种群
    def init_population(self):
        self.X = np.random.uniform(-self.W, self.W, (self.popu_size, self.dim))  # 范围内(-100, 100)随机采样，得到粒子位置(大小（20，30）)
        self.pbest = np.copy(self.X)    # 个体最优位置
        min_index = np.argmin(self.get_fitness(self.pbest))
        self.gbest = self.pbest[min_index]  # 全局最优位置
        self.V = np.random.uniform(-1, 1, (self.popu_size, self.dim))  # 范围内随机采样，得到粒子初始速度
    
    # 计算粒子群的适应度
    def get_fitness(self, X):
        fit = []
        for x in X:
            fit.append(self.fitness_fun(x))
        return np.array(fit)
    
    # 更新速度
    def update_velocity(self, i, j):
        # 根据公式更新速度
        new_v = self.w*self.V[i][j] + self.c1*np.random.rand()*(self.pbest[i][j]-self.X[i][j]) + \
            self.c2*np.random.rand()*(self.gbest[j]-self.X[i][j])
        
        # 限定速度范围
        if new_v > self.W:
            new_v = self.W
        if new_v < -self.W:
            new_v = -self.W
        self.V[i][j] = new_v

    # 更新最优位置
    def update_best_pos(self, i):
        # 个体
        if self.fitness_fun(self.X[i]) < self.fitness_fun(self.pbest[i]):
            self.pbest[i] = self.X[i]
        # 全局
        if self.fitness_fun(self.pbest[i]) < self.fitness_fun(self.gbest):
            self.gbest = self.pbest[i]
    
    # 速度更新 位置更新
    def update_velocity_escape_pos(self):
        for i in range(self.popu_size):
            for j in range(self.dim):
                self.update_velocity(i, j)
                # 更新粒子位置
                self.X[i][j] += self.V[i][j]

                # 限定位置范围
                if self.X[i][j] > self.W:
                    self.X[i][j] = self.W
                if self.X[i][j] < -self.W:
                    self.X[i][j] = -self.W

                # 更新最优位置    
                self.update_best_pos(i)
    
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
                # 更新速度 位置更新
                self.update_velocity_escape_pos()
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
    
