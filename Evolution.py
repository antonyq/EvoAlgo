import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


F = lambda x, y: 100*(y - x**2)**2 + (1 - x)**2
X_MIN, X_MAX = -1, 1
Y_MIN, Y_MAX = -1, 1
N = 20

class Evolution():
    def __init__(self):
        self.population = self.__eval_fitness__(np.array(np.random.uniform(low=X_MIN, high=X_MAX, size=(N, 2))))
        self.iteration = 0

    def __crossover__(self, x, y): 
        x = np.copy(x)
        point = np.random.randint(0, x.size, size=1)[0]
        for i in range(point):
            x[i] = y[i]
        return x

    def __mutate__(self, x, _min, _max, probability=0.3):
        x = np.copy(x)
        for i in range(x.size):
            if (np.random.rand() < probability):
                x[i] = np.random.uniform(_min, _max, (1,))
        return x

    def __select__(self, x, N=N):
        x = x[x[:,2].argsort()]
        x = x[:N]
        return x

    def __eval_fitness__(self, population):
        x, y = list(population.T[0]), list(population.T[1])
        f = np.array([[i] for i in list(map(F, x, y))])
        return np.append(population, f, axis=1)

    def __get_distinct_index__(self, number, arr, top_limit):
        for i in range(len(arr)):
            if arr[i] == number:
                if arr[i] + 1 < top_limit:
                    arr[i] += 1
                else:
                    arr[i] -= 1
        return arr

    def __stats__(self):
        transposed = self.population.T
        print(f'''
                {self.iteration} fitness
                min:{transposed[2].min()} x:{transposed[0][np.argmin(transposed[2])]} y:{transposed[1][np.argmin(transposed[2])]} 
                avg:{transposed[2].mean()} 
                max:{transposed[2].max()} x:{transposed[0][np.argmax(transposed[2])]} y:{transposed[1][np.argmin(transposed[2])]}
        ''')

    def __plot__(self):
        ax.clear()
        x, y = np.arange(X_MIN, X_MAX, 0.1), np.arange(Y_MIN, Y_MAX, 0.1)
        X, Y = np.meshgrid(x, y)
        Z = F(X, Y)
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim3d(X_MIN, X_MAX)
        ax.set_ylim3d(Y_MIN, Y_MAX)
        ax.view_init(30, self.iteration * 2)

    def evolutionary_algorithm(func):
        def algo(self, frame):
            self.iteration += 1
                
            func(self, frame)

            self.__plot__()
            self.__stats__()
            X, Y, Z = self.population.T[0], self.population.T[1], self.population.T[2]
            return ax.scatter(X, Y, Z)
        return algo
                

    @evolutionary_algorithm
    def genetic(self, frame):
        population_count = self.population.shape[0]
        gen_population = np.copy(self.population)
        for i in range(population_count):
            j = self.__get_distinct_index__(i, np.random.randint(0, population_count - 1, size=1), 1)[0]
            offspring = self.__crossover__(self.population[i], self.population[j])
            offspring = self.__mutate__(offspring, X_MIN, X_MAX)
            offspring[2] = F(offspring[0], offspring[1])
            gen_population = np.append(gen_population, [offspring], axis=0)
        self.population = self.__select__(gen_population)

    @evolutionary_algorithm
    def evolution_strategies(self, frame):
        d = np.sqrt(self.population.shape[0] + 1)
        sigma = np.random.random()
        for i in range(N):
            x = self.population[i] + sigma * np.random.normal(0, 1, size=(3, ))
            I = 1 if F(x[0], x[1]) < F(self.population[i][0], self.population[i][1]) else 0
            sigma = sigma * np.exp(I - 0.2)**(1/d)
            if I:
                self.population[i] = x

    @evolutionary_algorithm
    def differential_evolution(self, frame):
        MF = 0.8 # Mutation Force
        for i in range(N):
            rand_sample = np.random.random_integers(0, N - 1, size=3)
            k, m, n = self.__get_distinct_index__(i, rand_sample, N)
            C = self.population[k] + MF * (self.population[m] - self.population[n])
            offspring = self.__crossover__(self.population[i], C)
            offspring[2] = F(offspring[0], offspring[1])
            if offspring[2] < self.population[i][2]:
                self.population[i] = offspring
        
    @evolutionary_algorithm    
    def cooperative(self, frame):
        # np.random.shuffle(self.population)
        groups = np.array(np.split(self.population, 4))
        for n in range(groups.shape[0]):
            group = np.copy(groups[n])
            for i in range(groups.shape[1]):
                j = self.__get_distinct_index__(i, np.random.randint(0, groups.shape[1] - 1, size=1), 1)[0]
                offspring = self.__crossover__(group[i], group[j])
                offspring = self.__mutate__(offspring, X_MIN, X_MAX)
                offspring[2] = F(offspring[0], offspring[1])
                group = np.append(group, [offspring], axis=0)
            groups[n] = self.__select__(group, 5)
        self.population = groups.reshape(20, 3)

if __name__ == '__main__':
    evo = Evolution()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.hold(True)
    ani = FuncAnimation(fig, evo.evolution_strategies, N * 10, interval=1, blit=False)
    plt.show()