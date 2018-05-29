import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class Ecosystem():
    def __init__(self, organisms=20, dimensions=2, max_iterations=200):
        self.__max_iterations__ = max_iterations
        # Bohachevsky1
        self.x_skeleton = np.random.uniform(low=-100, high=100, size=(100, dimensions))
        self.x = np.random.uniform(low=-100, high=100, size=(organisms, dimensions) )
        self.f = lambda x: x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1]) + 0.7
        self.fitness = [self.f(xi) for xi in self.x]
        self.sk_fitness = [self.f(xi) for xi in self.x_skeleton]

    def __get_best_organism_number__(self):
        return self.fitness.index(np.min(self.fitness))

    def __get_random_organism_number__(self, except_i):
        j = except_i
        while (except_i == j):
            j = np.random.randint(0, self.x.size / 2)
        return j

    def mutualism(self, i):
        j = self.__get_random_organism_number__(i)
        xi, xj = self.x[i], self.x[j]

        mutual_vector = (xi + xj) / 2
        BF = np.random.randint(1, 3, size=(1, 2))
        
        xi_new = xi + np.random.rand()*(xi - mutual_vector*BF[0][0])
        xj_new = xj + np.random.rand()*(xi - mutual_vector*BF[0][1])
        
        self.x[i] = xi_new if self.f(xi_new) < self.f(self.x[i]) else self.x[i]
        self.x[j] = xj_new if self.f(xj_new) < self.f(self.x[j]) else self.x[j]

    def commensalism(self, i):
        j = self.__get_random_organism_number__(i)
        xi_new = self.x[i] + np.random.randint(-1,2)*(self.x[i] - self.x[j])
        self.x[i] = xi_new if self.f(xi_new) < self.f(self.x[i]) else self.x[i]

    def parasitism(self, i):
        j = self.__get_random_organism_number__(i)
        self.x[j] = self.x[i] if self.f(self.x[i]) < self.f(self.x[j]) else self.x[j]

    def run(self):
        i = 0
        while(i < self.x.size):
            i += 1
            Xbest_index = self.__get_best_organism_number__()
            print(i, Xbest_index, self.f(self.x[Xbest_index]))
            
            self.mutualism(Xbest_index)
            self.commensalism(Xbest_index)
            self.parasitism(Xbest_index)

            if i % 10 == 0:  
                self.show()
        plt.show()

    def show(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(self.x.T[0], self.x.T[1], self.fitness, color='red')
        ax.plot_trisurf(self.x_skeleton.T[0], self.x_skeleton.T[1], self.sk_fitness, cmap='cubehelix', linewidth=0.2)
        


def main():
    eco = Ecosystem(100, 2)
    eco.run()

if __name__ == '__main__':
    main()