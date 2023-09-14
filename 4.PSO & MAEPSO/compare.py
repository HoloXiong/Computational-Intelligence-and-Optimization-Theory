from Benchmark import Benchmark
from PSO import PSO
from MAEPSO import MAEPSO
import matplotlib.pyplot as plt
import numpy as np

# 画图
def draw_log_fitness(fit1, fit2, fun, epoch):
    plt.figure()
    fit1 = np.array(fit1)
    fit2 = np.array(fit2)
    plt.plot(np.log(fit1), label='PSO')
    plt.plot(np.log(fit2), label='MAEPSO')
    plt.xlabel('generation')
    plt.ylabel('Log(fitness)')
    plt.title(fun)
    plt.legend()
    # plt.show()
    plt.savefig('./results/' + fun + str(epoch) + '.png')

if __name__ == '__main__':
    print('Tablet PSO:')
    pso = PSO(Benchmark.Tablet)
    fit1 = pso.particle_evolution('Tablet')
    print('Tablet MAEPSO:')
    maepso = MAEPSO(Benchmark.Tablet)
    fit2 = maepso.particle_evolution('Tablet')
    draw_log_fitness(fit1, fit2, 'Tablet', 0)

    # print('Quadric PSO:')
    # pso = PSO(Benchmark.Quadric)
    # fit1 = pso.particle_evolution('Quadric')
    # print('Quadric MAEPSO:')
    # maepso = MAEPSO(Benchmark.Quadric)
    # fit2 = maepso.particle_evolution('Quadric')
    # draw_log_fitness(fit1, fit2, 'Quadric', 0)

    # print('Rosenbrock PSO:')
    # pso = PSO(Benchmark.Rosenbrock)
    # fit1 = pso.particle_evolution('Rosenbrock')
    # print('Rosenbrock MAEPSO:')
    # maepso = MAEPSO(Benchmark.Rosenbrock)
    # fit2 = maepso.particle_evolution('Rosenbrock')
    # draw_log_fitness(fit1, fit2, 'Rosenbrock', 0)

    # print('Griewank PSO:')
    # pso = PSO(Benchmark.Griewank)
    # fit1 = pso.particle_evolution('Griewank')
    # print('Griewank MAEPSO:')
    # maepso = MAEPSO(Benchmark.Griewank)
    # fit2 = maepso.particle_evolution('Griewank')
    # draw_log_fitness(fit1, fit2, 'Griewank', 0)

    # print('Rastrigin PSO:')
    # pso = PSO(Benchmark.Rastrigin)
    # fit1 = pso.particle_evolution('Rastrigin')
    # print('Rastrigin MAEPSO:')
    # maepso = MAEPSO(Benchmark.Rastrigin)
    # fit2 = maepso.particle_evolution('Rastrigin')
    # draw_log_fitness(fit1, fit2, 'Rastrigin', 0)

    # print('SchafferF7 PSO:')
    # pso = PSO(Benchmark.SchafferF7)
    # fit1 = pso.particle_evolution('SchafferF7')
    # print('SchafferF7 MAEPSO:')
    # maepso = MAEPSO(Benchmark.SchafferF7)
    # fit2 = maepso.particle_evolution('SchafferF7')
    # draw_log_fitness(fit1, fit2, 'SchafferF7', 0)