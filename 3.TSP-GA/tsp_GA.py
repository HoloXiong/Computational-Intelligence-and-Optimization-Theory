'''
holoxiong
实现模拟进化算（遗传算法）法解决旅行商问题
'''

# 读取数据
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
    plt.title('The Optimal Path of TSP by GA')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# 绘制适应度曲线
def draw_fitness(fitness):
    plt.figure(2)
    plt.plot(fitness)
    plt.title('The Fitness Curve')
    plt.ylabel('Fitness')
    plt.xlabel('Epoch')
    plt.show()

# 适应度函数，总路径短，适应度高
def get_path_distance(population, distance):
    popu_dists = []
    for path in population:
        total_distance = 0
        num = len(path)
        for i in range(num-1):
            total_distance += distance[path[i]-1][path[i+1]-1]
        # 由于是回路，加上最后一个城市回来的距离
        total_distance += distance[path[num-1]-1][path[0]-1]
        popu_dists.append(total_distance)
    return popu_dists

# 初始化种群 
def init_population(population_size):
    population = []
    for i in range(population_size):
        population.append(random.sample(range(1, 30), 29))    # 随机产生1-29的数
    return population

# Grefenstette编码
def grefen_encode(path):
    code = list(range(1, 30))
    encode_path = []
    for i in path:
        encode_path.append(code.index(i))    # 城市所在索引
        code.pop(code.index(i))              # 移除该城市
    return encode_path

# Grefenstette解码
def grefen_decode(path):
    code = list(range(1, 30))
    decode_path = []
    for i in range(len(path)):
        decode_path.append(code.pop(path[i]))      # 城市在code中索引，并移除
    return decode_path

# 选择
def select(population, distance):
    new_generation = []
    # 锦标赛选择
    # 进行20轮锦标赛，每轮从种群中选20人参赛，选适应度前5（距离小的路径）
    for i in range(20):
        group = []
        random.shuffle(population)  # 置乱种群
        for j in range(20):
            group.append(random.choice(population))
        # 适应度计算
        dist = get_path_distance(group, distance)
        # 排序，取出距离短的前五个个体
        winners = np.array(dist).argsort()   # 输出排序对应的索引
        winners = winners[0:5]
        for winner in winners:
            new_generation.append(group[winner])
    return new_generation

# 交叉
def crossover(population, crossover_rate):
    random.shuffle(population)                          # 置乱种群
    for i in range(len(population)): 
        population[i] = grefen_encode(population[i])    # 编码
    for i in range(0, len(population)-1, 2):
        if crossover_rate > random.random():
            # 父代
            father = population[i]    
            mother = population[i+1]

            # 产生两个边界随机数，交叉边界之间的片段
            left = random.randint(0, len(population[i])-1)
            right = random.randint(0, len(population[i])-1)
            if left > right:
                left, right = right, left
        
            # 中间不变，两边交换
            gene1 = mother[0:left] + father[left:right+1] + mother[right+1:]
            gene2 = father[0:left] + mother[left:right+1] + father[right+1:]
            population[i] = gene1
            population[i+1] = gene2
    for i in range(len(population)): 
        population[i] = grefen_decode(population[i])    # 解码
    return population

# 变异
def mutation(population, mutation_rate):
    for i in range(len(population)):
        if mutation_rate > random.random():
            parent = population[i]
            # 产生两个边界随机数，倒位变异边界之间的片段
            left = random.randint(0, len(population[i])-1)
            right = random.randint(0, len(population[i])-1)
            if left > right:
                left, right = right, left
            middle = parent[left:right+1]
            middle.reverse()
            gene = parent[0:left] + middle + parent[right+1:]
            population[i] = gene    
    return population

# 遗传算法
def GA():
    # 参数设置
    population_size = 100    # 种群空间大小
    crossover_rate = 0.9      # 交叉概率
    mutation_rate = 0.1       # 变异概率
    generation = 500          # 最大迭代次数

    # 城市数据导入
    distance, coordinate = load_data('bays29.tsp')

    # 初始化种群和初始种群距离
    population = init_population(population_size)    
    popu_dists = get_path_distance(population, distance)
    min_index = popu_dists.index(min(popu_dists))
    optimal_dist = popu_dists[min_index]
    optimal_path =  population[min_index]
    optimal_dist_list = []
    optimal_dist_list.append(optimal_dist)

    # 模拟进化
    iter = 0
    while iter < generation:
        iter += 1
        # 选择
        population = select(population, distance)
        # 交叉
        population = crossover(population, crossover_rate)
        # 变异
        population = mutation(population, mutation_rate)

        # 计算距离
        popu_dists = get_path_distance(population, distance)
        min_index = popu_dists.index(min(popu_dists))
        optimal_dist = popu_dists[min_index]
        optimal_path = population[min_index]
        optimal_dist_list.append(optimal_dist)
        print("epoch:{}\tmin_distance:{}".format(iter, optimal_dist))
    
    # 最终搜索结果
    print("min distance:{}".format(optimal_dist))
    print("optimal path:{}".format(optimal_path))

    # 画图
    draw_map(coordinate, optimal_path)
    draw_fitness(optimal_dist_list)

if __name__ == "__main__":
    GA()