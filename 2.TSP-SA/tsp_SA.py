'''
holoxiong
实现模拟退火算法解决旅行商问题
'''

# 读取数据
import copy
import math
import random
import matplotlib.pyplot as plt
import numpy as np

def load_data(filename):
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
def draw_map(coordinate, path):
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
    plt.title('The optimal path of TSP by SA')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# 计算路径的总长度
def get_path_distance(path, distance):
    total_distance = 0
    num = len(path)
    for i in range(num-1):
        total_distance += distance[path[i]][path[i+1]]
    # 由于是回路，加上最后一个城市回来的距离
    total_distance += distance[path[num-1]][path[0]]
    return total_distance

# 路径变换
def get_new_path(path):
    # 注意用copy，不能修改之前的path
    new_path = copy.copy(path)
    # 产生两个随机数，交换该位置的城市
    num = len(path)
    # 29个城市，索引范围0-28，起点设置为城市1
    m = random.randint(0, num-1)
    n = random.randint(0, num-1)
    new_path[m], new_path[n] = new_path[n], new_path[m]
    return new_path

# SA算法
def SA():    
    # 设置初始温度
    t = 100
    # 设置退火温度
    tn = 1
    # 设置温度衰减率
    a = 0.98
    # 设置Markov链长度
    len_mar = 10000

    # 数据读取
    distance, coordinate = load_data('bays29.tsp')
    # 初始路径解
    now_path = random.sample(range(0, 29), 29)
    now_distance = get_path_distance(now_path, distance)
    # 最优路径
    opt_path = now_path
    # 最短距离
    opt_distance = now_distance

    # 模拟退火
    # 温度迭代
    while t > tn:
        # Markov链
        for i in range(len_mar):
            # 路径变换
            new_path = get_new_path(now_path)
            # 计算新的路径总距离
            new_distance = get_path_distance(new_path, distance)
            # 距离之差
            delta = new_distance - now_distance

            # 根据Metropolis算法确定是否接受新解
            # 得到优解,接受
            if delta <= 0:
                now_path = new_path
                now_distance = new_distance
                # 总距离缩短则更新
                if opt_distance > now_distance:
                    opt_distance = now_distance
                    opt_path = now_path
            # 得到较差解，计算判定是否接受
            else:
                p = math.exp(-delta / t)
                # 产生一个随机概率, p大则接受差解
                if p > random.random():
                    now_distance = new_distance
                    now_path = new_path
        # 降温，退火操作
        t = t*a
    
    # 输出模拟退火找到的最短路径
    opt_path = [i+1 for i in opt_path]
    real_path = [1, 28, 6, 12, 9, 5, 26, 29, 3, 2, 20, 10, 4, 15, 18, 17, 14, 22, 11, 19, 25, 7, 23, 27, 8, 24, 16, 13, 21]
    print(opt_path)
    print("Total distance:{}".format(opt_distance))

    # 对比真正最短路径
    print(real_path)
    real_path = [i-1 for i in real_path]
    real_distance = get_path_distance(real_path, distance)
    print("Real distance:{}".format(real_distance))

    # 绘图
    draw_map(coordinate, opt_path)

# 主函数
if __name__ == "__main__":
    print('Processing......')
    SA()