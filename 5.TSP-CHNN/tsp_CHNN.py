import random
import matplotlib.pyplot as plt
import numpy as np
import copy

"""
利用HopField网络(CHNN)解决TSP问题
author: holoxiong
date:2022.11.26
"""

class CHNN:
    def __init__(self, epochs, lr, U0, A, D, num) -> None:
        self.epochs = epochs    # 迭代次数
        self.lr = lr  # 学习率  
        self.U0 = U0  # 初始电压
        self.A = A  # 两个网络参数
        self.D = D
        self.num = num  # 城市数量
        self.distance, self.coordinate = self.load_data('bays29.tsp')  # 读取城市之间的距离和坐标
        self.V = self.init_V()  # 初始化换位矩阵
        self.U = self.init_U()  # 初始化电压矩阵

    # 获取距离和坐标
    def load_data(self, filename):
        # 距离
        distance = []
        # 坐标
        coordinate = []
        with open(filename) as f:
            line = f.readline()
            index = 1
            while line:
                if index >= 9 and index <= 37:
                    # 读取并且切分
                    data = line.strip('\n').split(' ')
                    # 去除空格
                    data = list(filter(lambda x : x, data))
                    # 转为int
                    data = list(map(float, data))
                    distance.append(data)
                if index >= 39 and index <= 67:
                    data = line.strip('\n').split(' ')
                    data = list(filter(lambda x : x, data))
                    data = list(map(float, data))
                    coordinate.append(data) 
                line = f.readline()
                index += 1
        return distance, coordinate

    # 绘制这29个坐标的路径图
    def draw_map(self, coordinate, path):
        x = []
        y = []
        index = []
        for city in path:
            index.append(int(coordinate[city-1][0]))
            x.append(coordinate[city-1][1])
            y.append(coordinate[city-1][2])
        x.append(coordinate[path[0]-1][1])
        y.append(coordinate[path[0]-1][2])
        fig, ax = plt.subplots()
        ax.plot(x, y, marker='$\\bigotimes$')
        for i in range(29):
            ax.annotate(index[i], (x[i], y[i]))
        plt.title('The optimal path of TSP by CHNN')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('opt_path.png')

    # 计算路径的总长度
    def get_path_distance(self, path, distance):
        total_distance = 0
        num = len(path)
        for i in range(num-1):
            total_distance += distance[path[i]][path[i+1]]
        # 由于是回路，加上最后一个城市回来的距离
        total_distance += distance[path[num-1]][path[0]]
        return total_distance
    
    # 标准化距离矩阵
    def normalize_distance(self):
        dist = np.array(self.distance)
        max_dist = np.max(dist)
        for i in range(self.num):
            for j in range(self.num):
                self.distance[i][j] = self.distance[i][j] / max_dist
    
    # 初始化换位矩阵
    def init_V(self):
        path = random.sample(range(0, 29), 29)  # 随机初始化一条路径解
        V = np.zeros((self.num, self.num))
        for i in range(self.num):
            V[path[i]][i] = 1
        return V
    
    # 初始化电压矩阵
    def init_U(self):
        return 0.5*self.U0*np.log(self.num-1) + 2*(np.random.random((self.num, self.num)))-1
    
    # 根据公式计算du/dt
    def calculate_du_dt(self, V):
        du_dt = np.zeros((self.num, self.num))
        for x in range(self.num):
            for i in range(self.num):
                sum1 = 0
                sum2 = 0
                sum3 =0
                # 对V行求和
                for j in range(self.num):
                    sum1 = sum1 + V[x][j]
                # 对V列求和
                for y in range(self.num):
                    sum2 = sum2 + V[y][i]
                    if i < self.num-1:
                        sum3  = sum3 + self.distance[x][y] * V[y][i+1]
                    else:
                        sum3 = sum3 + self.distance[x][y] * V[y][0]
                du_dt[x][i] = -self.A * (sum1-1) - self.A * (sum2 - 1) - self.D * sum3
        return du_dt
    
    # 更新电压U
    def update_U(self, U, du_dt):
        new_U = U +  du_dt * self.lr
        return new_U
    
    # 更新换位矩阵V
    def update_V(self, U):
        new_V =  0.5 * (1 + np.tanh(U / self.U0))
        return new_V

    # 根据公式计算能量
    def calculate_energy(self, V):
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum1 = np.sum(np.sum(V, axis=0)**2)
        sum2 = np.sum(np.sum(V, axis=1)**2)
        for x in range(self.num):
            for y in range(self.num):
                for i in range(self.num):
                    if i < self.num-1:
                        sum3 = sum3 + V[x][i] * self.distance[x][y] * V[x][i] * V[y][i+1]
                    else:
                        sum3 = sum3 + V[x][i] * self.distance[x][y] * V[x][i] * V[y][0]
        energy = self.A * sum1 * 0.5 + self.A * sum2 * 0.5 + self.D * sum3 * 0.5
        return energy
    
    # 由V获得路径解
    def get_path(self, V):
        path = []
        for i in range(self.num):
            max_v = np.max(V[:, i])  # 获得每一列中概率最大的值的索引
            for j in range(self.num):
                if V[j][i] == max_v:
                    path.append(j)
                    break
        return path

    # 网络构建与更新
    def train(self):
        print('Processing......')
        distance = copy.deepcopy(self.distance)
        self.normalize_distance()
        U = self.U
        V  = self.V
        energy_list = []
        dist_list = []
        for epoch in range(1, self.epochs+1):
            du_dt = self.calculate_du_dt(V)
            U = self.update_U(U, du_dt)
            V = self.update_V(U)
            energy = self.calculate_energy(V)
            energy_list.append(energy)
            path = self.get_path(V)
            if len(np.unique(path)) == self.num:
                dist = self.get_path_distance(path, distance)
                dist_list.append(dist)
                print('epoch{}:energy:{}\tpath:{}\tdistance:{}'.format(epoch, energy, path, dist))
        # 绘制最终路径图
        self.draw_map(self.coordinate, path)
        
        # 绘制能量曲线
        plt.figure()
        plt.plot(energy_list)
        plt.xlabel('Epoch')
        plt.ylabel('Energy')
        plt.title('Energy Curve of CHNN')
        plt.savefig('energy.png')

        # 绘制距离曲线
        plt.figure()
        plt.plot(dist_list)
        plt.xlabel('Path')
        plt.ylabel('Total Distance')
        plt.title('Distance Curve of CHNN')
        plt.savefig('distance.png')

if __name__ == "__main__":
    chnn = CHNN(10000, 0.05, 0.02 ,1.5, 0.5, 29)
    chnn.train()