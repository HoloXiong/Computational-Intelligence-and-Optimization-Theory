import math
import numpy as np

class Benchmark:
    """
    实现6个Benchmark函数
        单模态：Tablet, Quadric, Rosenbrock
        多模态：Griewank, Rastrigin, Schaffer F7
    author:holoxiong@stu.xmu.edu.cn
    date:2022.11.19
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    # Tablet函数
    def Tablet(x):
        left = np.power(10, 6) * np.square(x[0])
        right = np.sum(np.square(x[1:]))
        return left + right

    @staticmethod
    # Quadric函数
    def Quadric(x):
        sum = 0
        for i in range(len(x)):
            sum += np.square(np.sum(x[0:i+1]))  # 0-i求和
        return sum

    @staticmethod
    # Rosenbrock函数 note:参考文献中有误
    def Rosenbrock(x):
        sum = 0
        for i in range(len(x)-1):
            left = 100 * np.square(x[i+1]-np.square(x[i]))
            right = np.square(x[i]-1)
            sum += left + right
        return sum
    
    @staticmethod
    # Griewank函数
    def Griewank(x):
        left = np.sum(np.square(x)) / 4000
        mul_sum = 1
        for i in range(len(x)):
            mul_sum = mul_sum * np.cos(x[i]/np.sqrt(i+1))  # 注意不能根号0
        return left - mul_sum + 1

    @staticmethod
    # Rastrigin函数
    def Rastrigin(x):
        A = 10
        return np.sum(np.square(x) - A*np.cos(2*np.pi*x) + A)

    @staticmethod
    # SchafferF7函数
    def SchafferF7(x):
        return sum([(x[i] ** 2 + x[i + 1] ** 2) ** 0.25 for i in range(len(x) - 1)]) + \
           math.sin((50 * sum([(x[i] ** 2 + x[i + 1] ** 2) ** 0.1 for i in range(len(x) - 1)])) ** 2) + 1


if __name__ == '__main__':
    # print(np.power(2, 0.25))
    # print(2**0.25)
    #w = np.random.uniform(-100, 100, (20, 30))
    # print(np.random.randn(5))
    x = np.array([1,1,1,1])
    p = [1,2,3]
    p = np.array(p)
    p[p>1] = 9
    print(p)

    
