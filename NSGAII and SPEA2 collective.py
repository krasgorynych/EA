import math
import random
# matplotlib.pyplot as plt
import numpy as np


def func_all(vect, k):
    if k == 0:
        return 4*vect[0]**2 +  4*vect[1] **2
    if k == 1:
        return (vect[0] - 5)**2 +  (vect[1] - 5)**2
'''   
def func_all(vect, k):
    if k == 0:
        return 1 - math.exp(-1*((vect[0] - 1 / math.sqrt(3))**2 + (vect[1] - 1 / math.sqrt(3))**2 + (vect[2] - 1 / math.sqrt(3))**2))
    if k == 1:
        return 1 - math.exp(-1*((vect[0] + 1 / math.sqrt(3))**2 + (vect[1] + 1 / math.sqrt(3))**2 + (vect[2] + 1 / math.sqrt(3))**2))      
'''
def generational_distance(func, massiv_one, massiv_two, number_criteria):
    massiv_fitness_one = [[0]*number_criteria for i in range(len(massiv_one))]
    massiv_fitness_two = [[0]*number_criteria for i in range(len(massiv_two))]
    massiv_distance = [[0]*len(massiv_two) for i in range(len(massiv_one))]
    for i in range(len(massiv_one)):
        massiv_fitness_one[i] = [func(massiv_one[i], j) for j in range(number_criteria)]
    for i in range(len(massiv_two)):
        massiv_fitness_two[i] = [func(massiv_two[i], j) for j in range(number_criteria)]
    for i in range(len(massiv_one)):
        for j in range(len(massiv_two)):
            for k in range(number_criteria):
                massiv_distance[i][j] =  massiv_distance[i][j] + (massiv_fitness_one[i][k] - massiv_fitness_two[j][k])**2
            massiv_distance[i][j] = math.sqrt(massiv_distance[i][j])
    vector_min_distance = [min(massiv_distance[i])**2 for i in range(len(massiv_one))]
    GD = math.sqrt(sum(vector_min_distance)) / len(massiv_one)
    return GD
    
def spasing(func, massiv_one, number_criteria):
    massiv_fitness_one = [[0]*number_criteria for i in range(len(massiv_one))]
    massiv_distance = [[0]*len(massiv_one) for i in range(len(massiv_one))]
    for i in range(len(massiv_one)):
        massiv_fitness_one[i] = [func(massiv_one[i], j) for j in range(number_criteria)]
    for i in range(len(massiv_one)):
        for j in range(len(massiv_one)):
            if i != j:
                for k in range(number_criteria):
                    massiv_distance[i][j] =  massiv_distance[i][j] + math.fabs(massiv_fitness_one[i][k] - massiv_fitness_one[j][k])
    k = [massiv_distance[i].pop(massiv_distance[i].index(0)) for i in range(len(massiv_one))]            
    vector_min_distance = [min(massiv_distance[i]) for i in range(len(massiv_one))]
    middle = sum(vector_min_distance) / len(massiv_one)
    help_vector = [(middle - vector_min_distance[i])**2 for i in range(len(massiv_one))]
    S = math.sqrt(sum(help_vector) / (len(massiv_one) - 1))
    return S
 
class GA():
    def __init__(self, matrix_border, accuracy, number_individ, number_population, value_mutation, size_tour_selection):
        self.matrix_border = matrix_border #матрица границ
        self.accuracy = accuracy #точность
        self.number_individ = number_individ #количество индивидов
        self.number_population = number_population #количество популяций
        self.value_mutation = value_mutation #значение мутации
        self.size_tour_selection = size_tour_selection #размер турнира
        self.space_dimension = matrix_border.shape[0] #размерность пространства, количество элементов вектора х
        self.vector_size_chrom, self.size_chrom = self.calcul_size_chrom() #размер хромосомы
        self.matrix_genotype = [[0] * self.size_chrom for i in range(number_individ)]
        self.matrix_phenotype = [[0] * self.space_dimension for i in range(number_individ)]
        self.matrix_fitness = [[0] for i in range(number_individ)]
    
    def calcul_size_chrom(self):
        vector_size_chrom = [0 for i in range(self.space_dimension)]
        size_chrom = 0
        for i in range(self.space_dimension):
            a = (self.matrix_border[i][1] - self.matrix_border[i][0]) / self.accuracy
            vector_size_chrom[i] = math.floor(math.log10(a) / math.log10(2) + 1)
            size_chrom = size_chrom + vector_size_chrom[i]
        return vector_size_chrom, size_chrom
    
    def randomizer(self):    
        for i in range(self.number_individ):
            self.matrix_genotype[i] = [random.randint(0, 1) for k in range(self.size_chrom)]
        return self.matrix_genotype
    
    def conver_genotype_phenotype(self):
        for i in range(self.number_individ):
            for j in range(self.space_dimension):
                help1 = sum(self.vector_size_chrom[0:j])
                help2 = sum(self.vector_size_chrom[0:(j+1)])
                x = list(self.matrix_genotype[i][help1:help2])
                self.matrix_phenotype[i][j] = self.matrix_border[j][0] + int((''.join([str(k) for k in x])), 2) * self.accuracy
        return self.matrix_phenotype
   
    def fitness_func(self, func, k):
        for i in range(self.number_individ):
            self.matrix_fitness[i] = func(self.matrix_phenotype[i], k)
        return self.matrix_fitness
    
    def tour_selection(self):
        help_matrix_genotype = [[0] * self.size_chrom for i in range(self.number_individ)]
        for i in range(self.number_individ):
            list_random_1 = [random.randint(0, (self.number_individ - 1)) for k in range(self.size_tour_selection)]
            list_random_2 = [self.matrix_fitness[k] for k in list_random_1]
            help_matrix_genotype[i] = self.matrix_genotype[list_random_1[list_random_2.index(min(list_random_2))]] #max или min зависит от задачи
        self.matrix_genotype = help_matrix_genotype
        return self.matrix_genotype
    
    def single_recombination(self):
        help_matrix_genotype = [[0] * self.size_chrom for i in range(self.number_individ)]
        for i in range(self.number_individ):
            n_1 = random.randint(0, (self.number_individ-1))
            n_2 = random.randint(0, (self.number_individ-1))
            point = random.randint(1, (self.size_chrom - 1))
            help_matrix_genotype[i] = self.matrix_genotype[n_1][0:point] + self.matrix_genotype[n_2][point:self.size_chrom]
        self.matrix_genotype = help_matrix_genotype
        return self.matrix_genotype
   
    def double_recombination(self):
        help_matrix_genotype = [[0] * self.size_chrom for i in range(self.number_individ)]
        for i in range(0):
            n_1 = random.randint(0, (self.number_individ-1))
            n_2 = random.randint(0, (self.number_individ-1))
            point = [random.randint(1, (self.size_chrom - 1)) for k in range(2)]
            help_matrix_genotype[i] = self.matrix_genotype[n_1][0:min(point)] + self.matrix_genotype[n_2][min(point):max(point)] + self.matrix_genotype[n_1][max(point):self.size_chrom]
            help_matrix_genotype[i+1] = self.matrix_genotype[n_2][0:min(point)] + self.matrix_genotype[n_1][min(point):max(point)] + self.matrix_genotype[n_2][max(point):self.size_chrom]
        return help_matrix_genotype
    
    def equal_recombination(self):
        help_matrix_genotype = [[0] * self.size_chrom for i in range(self.number_individ)]
        for i in range(0, (self.number_individ-1), 2):
            n_1 = random.randint(0, (self.number_individ-1))
            n_2 = random.randint(0, (self.number_individ-1))
            help_matrix_genotype[i][0] = self.matrix_genotype[n_1][0]
            help_matrix_genotype[i+1][0] = self.matrix_genotype[n_2][0]
            for j in range(1, self.size_chrom):
                if j%2==0:
                    help_matrix_genotype[i][j] += self.matrix_genotype[n_1][j]
                    help_matrix_genotype[i+1][j] += self.matrix_genotype[n_2][j]
                else:
                    help_matrix_genotype[i][j] += self.matrix_genotype[n_2][j]
                    help_matrix_genotype[i+1][j] += self.matrix_genotype[n_1][j]
        self.matrix_genotype = help_matrix_genotype
        return self.matrix_genotype
    
    def mutation(self):
        help_matrix_genotype = [[0] * self.size_chrom for i in range(self.number_individ)]
        for i in range(self.number_individ):
            for j in range(self.size_chrom):
                value = random.randint(0, 1000) * 0.001
                if value < self.value_mutation:
                    help_matrix_genotype[i][j] = abs(self.matrix_genotype[i][j] - 1)
                else:
                    help_matrix_genotype[i][j] = self.matrix_genotype[i][j]
        self.matrix_genotype = help_matrix_genotype
        return self.matrix_genotype

class NSGAII(GA):
    def __init__(self, matrix_border, accuracy, number_individ, number_population, value_mutation, size_tour_selection, number_criteria):
        super().__init__(matrix_border, accuracy, number_individ, number_population, value_mutation, size_tour_selection)
        self.number_criteria = number_criteria
        self.matrix_rank = [0 for i in range(self.number_individ)]
        self.matrix_front = []
        self.vector_crowding_distance = [0 for k in range(self.number_individ)]
        
    def domination(self, func, p, q):
        count = 0
        for i in range(self.number_criteria):
            if func(self.matrix_phenotype[p], i)<=func(self.matrix_phenotype[q], i):
                count = count + 1
            elif func(self.matrix_phenotype[q], i)<=func(self.matrix_phenotype[p], i):
                count = count - 1
        return count
    
    def non_dominated_sort(self, func):
        massiv_front = []
        massiv_front_one = []
        s_p_all = []
        n_p_all = []
        for i in range(self.number_individ):
            s_p = []
            n_p = 0
            for j in range(self.number_individ):
                if j!=i:
                    if self.domination(func, i, j) == self.number_criteria:
                        s_p.append(j)
                    elif self.domination(func, i, j) == (-1*self.number_criteria):
                        n_p = n_p + 1
            if n_p == 0:
                self.matrix_rank[i] = 1
                massiv_front_one.append(i)
            s_p_all.append(s_p)
            n_p_all.append(n_p)
        massiv_front.append(massiv_front_one)
        i = 0
        while massiv_front[i] != []:
            Q = []
            for j in range(len(massiv_front[i])):
                for k in range(len(s_p_all[massiv_front[i][j]])):
                    n_p_all[s_p_all[massiv_front[i][j]][k]] = n_p_all[s_p_all[massiv_front[i][j]][k]] - 1
                    if n_p_all[s_p_all[massiv_front[i][j]][k]] == 0:
                        self.matrix_rank[s_p_all[massiv_front[i][j]][k]] = i + 1
                        Q.append(s_p_all[massiv_front[i][j]][k])
            massiv_front.append(Q)
            i = i + 1
        del massiv_front[-1]
        self.matrix_front = massiv_front
        for i in range(0, len(massiv_front)):
            for j in range(len(massiv_front[i])):
                self.matrix_rank[massiv_front[i][j]] = i+1
        return self.matrix_rank, self.matrix_front
    
    def crowing_distance(self, func):
        for i in range(len(self.matrix_front)):
            matrix_crowding_distance_i = [[k, 0., 0.] for k in self.matrix_front[i]]
            for m in range(self.number_criteria):
                for b in range(len(matrix_crowding_distance_i)):
                    matrix_crowding_distance_i[b][2] = func(self.matrix_phenotype[matrix_crowding_distance_i[b][0]], m)
                matrix_crowding_distance_i = sorted(matrix_crowding_distance_i, key=lambda matrix_crowding_distance_i: matrix_crowding_distance_i[2])
                matrix_crowding_distance_i[0][1] = matrix_crowding_distance_i[-1][1] = math.inf
                for k in range(1, (len(matrix_crowding_distance_i) - 1)):
                    if max(matrix_crowding_distance_i)[2] != min(matrix_crowding_distance_i)[2]:
                        matrix_crowding_distance_i[k][1] = matrix_crowding_distance_i[k][1] + (matrix_crowding_distance_i[k+1][2] - matrix_crowding_distance_i[k-1][2]) / (max(matrix_crowding_distance_i)[2] - min(matrix_crowding_distance_i)[2])
                    else:
                        matrix_crowding_distance_i[k][1] = matrix_crowding_distance_i[k][1] + 0
            for j in range(len(matrix_crowding_distance_i)):
                self.vector_crowding_distance[(matrix_crowding_distance_i[j][0])] = matrix_crowding_distance_i[j][1]
        return self.vector_crowding_distance
    
    def binary_tour_selection(self):
        help_matrix_genotype = [[0] * self.size_chrom for i in range(self.number_individ)]
        for i in range(self.number_individ):
            first_ind = random.randint(0, (self.number_individ - 1))
            second_ind = random.randint(0, (self.number_individ - 1))
            while second_ind == first_ind: second_ind = random.randint(0, (self.number_individ - 1))
            if self.matrix_rank[first_ind] < self.matrix_rank[second_ind]:
                help_matrix_genotype[i] = self.matrix_genotype[first_ind]
            elif self.matrix_rank[first_ind] > self.matrix_rank[second_ind]:
                help_matrix_genotype[i] = self.matrix_genotype[second_ind]
            elif self.matrix_rank[first_ind] == self.matrix_rank[second_ind] and self.vector_crowding_distance[first_ind] > self.vector_crowding_distance[second_ind]:
                help_matrix_genotype[i] = self.matrix_genotype[first_ind]
            else: help_matrix_genotype[i] = self.matrix_genotype[second_ind]
        self.matrix_genotype = help_matrix_genotype
        return self.matrix_genotype

class SPEA2(GA):
    def __init__(self, matrix_border, accuracy, number_individ, number_population, value_mutation, size_tour_selection, number_criteria, number_individ_archive):
        super().__init__(matrix_border, accuracy, number_individ, number_population, value_mutation, size_tour_selection)
        self.number_criteria = number_criteria
        self.number_individ_archive = number_individ_archive
        self.matrix_genotype_archive = []
        self.matrix_phenotype_archive = []
        self.matrix_fitness_archive = []
        
    def domination(self, func, p, q):
        count = 0
        for i in range(self.number_criteria):
            if func(p, i)<=func(q, i):
                count = count + 1
            elif func(q, i)<=func(p, i):
                count = count - 1
        return count
    
    def euclidean_metric(self, func, a, b):
        d = 0
        for i in range(self.number_criteria):
            d = d + (func(a, i) - func(b, i))**2
        d = math.sqrt(d)
        return d
    
    def criteria1(self, len_archive, matrix_d, i, j):
        count = 0
        for k in range(1, len_archive):
            if matrix_d[i][k] == matrix_d[j][k]:
                count = count + 1
        if count == (len_archive - 1):
            return True
        else: 
            return False
        
    def criteria2(self, len_archive, matrix_d, i, j):
        for k in range(1, len_archive):
            count = 0
            for l in range(1, k):
                if matrix_d[i][k] < matrix_d[j][k] and matrix_d[i][l] == matrix_d[j][l]:
                    count = count + 1
            if k == (count + 1):
                return True
        return False
    
    def calcul_power(self, func):
        help_vector_power = [0]*(self.number_individ+self.number_individ_archive)
        help_matrix_domination = [[0] for i in range(self.number_individ+self.number_individ_archive)]
        for i in range(self.number_individ+self.number_individ_archive):
            if i < self.number_individ:
                for j in range(self.number_individ):
                    domination = self.domination(func, self.matrix_phenotype[i], self.matrix_phenotype[j])
                    if i != j and domination == self.number_criteria:
                        help_vector_power[i] = help_vector_power[i] + 1
                    elif i != j and abs(domination) == self.number_criteria:
                        help_matrix_domination[i].append(j)
                for j in range(self.number_individ_archive):
                    domination = self.domination(func, self.matrix_phenotype[i], self.matrix_phenotype_archive[j])
                    if domination == self.number_criteria:
                        help_vector_power[i] = help_vector_power[i] + 1     
                    elif abs(domination) == self.number_criteria:
                        help_matrix_domination[i].append(j+self.number_individ)
            else:
                for j in range(self.number_individ):
                    domination = self.domination(func, self.matrix_phenotype_archive[i-self.number_individ], self.matrix_phenotype[j])
                    if domination == self.number_criteria:
                        help_vector_power[i] = help_vector_power[i] + 1
                    elif abs(domination) == self.number_criteria:
                        help_matrix_domination[i].append(j)
                for j in range(self.number_individ_archive):
                    domination = self.domination(func, self.matrix_phenotype_archive[i-self.number_individ], self.matrix_phenotype_archive[j])
                    if i != j and domination == self.number_criteria:
                        help_vector_power[i] = help_vector_power[i] + 1   
                    elif abs(domination) == self.number_criteria:
                        help_matrix_domination[i].append(j+self.number_individ)
        help_matrix_domination = [i[1::] for i in help_matrix_domination]
        return help_vector_power, help_matrix_domination
    
    def calcul_raw_fitness(self, func):
        vector_power, matrix_domination = self.calcul_power(func)
        all_ind = self.number_individ+self.number_individ_archive
        vector_raw_fitness = [0]*all_ind
        for i in range(all_ind):
            vector_raw_fitness[i] = sum([vector_power[k] for k in matrix_domination[i]])
        return vector_raw_fitness
        
    def calcul_density(self, func):
        all_ind = self.number_individ+self.number_individ_archive
        k = math.floor(math.sqrt(all_ind))
        matrix_distance = [[0]*(all_ind) for i in range(all_ind)]
        for i in range(all_ind):
            for j in range(all_ind):
                if i != j:
                    if i < self.number_individ and j < self.number_individ:
                        matrix_distance[i][j] = self.euclidean_metric(func, self.matrix_phenotype[i], self.matrix_phenotype[j])
                    elif i < self.number_individ and j >= self.number_individ:
                        matrix_distance[i][j] = self.euclidean_metric(func, self.matrix_phenotype[i], self.matrix_phenotype_archive[j-self.number_individ])
                    elif i >= self.number_individ and j < self.number_individ:
                        matrix_distance[i][j] = self.euclidean_metric(func, self.matrix_phenotype_archive[i-self.number_individ], self.matrix_phenotype[j])
                    else:
                        matrix_distance[i][j] = self.euclidean_metric(func, self.matrix_phenotype_archive[i-self.number_individ], self.matrix_phenotype_archive[j-self.number_individ])
            matrix_distance[i].sort()
        vector_distance = [i[k-1] for i in matrix_distance]
        vector_density = [ (1 / (i + 2)) for i in vector_distance]
        return vector_density
    
    def spea_fitness_func(self, func):
        vector_raw_fitness = self.calcul_raw_fitness(func)
        vector_density = self.calcul_density(func)
        help_vector_fitness = [vector_raw_fitness[i]+vector_density[i] for i in range(self.number_individ+self.number_individ_archive)]
        self.matrix_fitness = help_vector_fitness[0:self.number_individ]
        self.matrix_fitness_archive = help_vector_fitness[self.number_individ::]
        return self.matrix_fitness, self.matrix_fitness_archive

    def form_new_archive(self, func):
        help_genotype_new_archive = []
        help_phenotype_new_archive = []
        help_fitness_new_archive = []
        vector_raw_fitness = self.calcul_raw_fitness(func)
        for i in range(self.number_individ+self.number_individ_archive):
            if i < self.number_individ and vector_raw_fitness[i] == 0:
                help_genotype_new_archive.append(self.matrix_genotype[i])
                help_phenotype_new_archive.append(self.matrix_phenotype[i])
                help_fitness_new_archive.append(self.matrix_fitness[i])
            elif i >= self.number_individ and vector_raw_fitness[i] == 0:
                help_genotype_new_archive.append(self.matrix_genotype_archive[i-self.number_individ])
                help_phenotype_new_archive.append(self.matrix_phenotype_archive[i-self.number_individ])
                help_fitness_new_archive.append(self.matrix_fitness_archive[i-self.number_individ])
        self.matrix_genotype_archive = help_genotype_new_archive
        self.matrix_phenotype_archive = help_phenotype_new_archive
        self.matrix_fitness_archive = help_fitness_new_archive
        return self.matrix_phenotype_archive
    
    def clustering(self, func, len_archive):
        k = math.floor(math.sqrt(self.number_individ_archive))
        matrix_distance = [[0]*len_archive for i in range(len_archive)]
        for i in range(len_archive):
            for j in range(len_archive):
                if i != j:
                    matrix_distance[i][j] = self.euclidean_metric(func, self.matrix_phenotype_archive[i], self.matrix_phenotype_archive[j])
            matrix_distance[i].sort()
        i = 0
        while  len_archive > self.number_individ_archive:
            k = 0
            for j in range(len_archive):
                if i != j:
                    if self.criteria1(len_archive, matrix_distance, i, j) == True or self.criteria2(len_archive, matrix_distance, i, j) == True:
                        k = k + 1
            if k == len_archive - 1:
                len_archive = len_archive - 1
                self.matrix_genotype_archive.pop(i)
                self.matrix_fitness_archive.pop(i)
                self.matrix_phenotype_archive.pop(i)
                matrix_distance.pop(i)
                perem = [matrix_distance[l].pop(i) for l in range(len_archive)]
            if (i + 1) < len_archive:
                i = i + 1
            elif i == len_archive:
                i = i - 1
        return self.matrix_phenotype_archive
             
    def add_form_new_archive(self):
        help_vector_fitness = [[self.matrix_fitness[i], i] for i in range(self.number_individ) if self.matrix_fitness[i] != 0]
        while len(self.matrix_genotype_archive) < self.number_individ_archive:
            perem = help_vector_fitness.index(min(help_vector_fitness))
            self.matrix_genotype_archive.append(self.matrix_genotype[help_vector_fitness[perem][1]])
            self.matrix_phenotype_archive.append(self.matrix_phenotype[help_vector_fitness[perem][1]])
            self.matrix_fitness_archive.append(self.matrix_fitness[help_vector_fitness[perem][1]])
        return self.matrix_phenotype_archive
    
    def spea_tour_selection(self):
        help_matrix_genotype = [[0] * self.size_chrom for i in range(self.number_individ)]
        help_vector_fitness = self.matrix_fitness + self.matrix_fitness_archive
        help_matrix_all_genotype = self.matrix_genotype + self.matrix_genotype_archive
        for i in range(self.number_individ):
            list_random_1 = [random.randint(0, (self.number_individ + self.number_individ_archive - 1)) for k in range(self.size_tour_selection)]
            list_random_2 = [help_vector_fitness[k] for k in list_random_1]
            help_matrix_genotype[i] = help_matrix_all_genotype[list_random_1[list_random_2.index(min(list_random_2))]] #max или min зависит от задачи
        self.matrix_genotype = help_matrix_genotype
        return self.matrix_genotype
  
a = np.array([[0, 5], [0, 3]])
#a = np.array([[-4, 4], [-4, 4], [-4, 4]])

number_iteration = 10
GD = [0]*number_iteration
S = [0]*number_iteration

number_half_individ = 250
number_half_individ_archive = int(number_half_individ / 2)
number_migration = number_half_individ
koef = 2
number_all_population = 100
number_piece_population = int(number_all_population / koef)

for iter in range(number_iteration):
    objgansga = NSGAII(a, 0.1, number_half_individ, number_piece_population, 0.1, 2, 2)
    objgaspea = SPEA2(a, 0.1, number_half_individ, number_piece_population, 0.1, 2, 2, number_half_individ_archive)
    objgaspea.randomizer()
    objgansga.randomizer()
    
    for iter_2 in range(koef):
        objgaspea.conver_genotype_phenotype()
        objgaspea.tour_selection()
        objgaspea.matrix_genotype_archive = [objgaspea.matrix_genotype[i].copy() for i in range(objgaspea.number_individ_archive)]
        objgaspea.matrix_phenotype_archive = [objgaspea.matrix_phenotype[i].copy() for i in range(objgaspea.number_individ_archive)]
        for i in range(number_piece_population):
            objgansga.conver_genotype_phenotype()
            objgansga.non_dominated_sort(func_all)
            objgansga.crowing_distance(func_all)
            objgansga.binary_tour_selection()
            objgansga.mutation()
            objgaspea.conver_genotype_phenotype()
            objgaspea.spea_fitness_func(func_all)
            objgaspea.form_new_archive(func_all)
            if len(objgaspea.matrix_genotype_archive) > objgaspea.number_individ_archive:
                objgaspea.clustering(func_all, len(objgaspea.matrix_genotype_archive))
            elif len(objgaspea.matrix_genotype_archive) < objgaspea.number_individ_archive:
                objgaspea.add_form_new_archive()
            objgaspea.spea_tour_selection()
            objgaspea.single_recombination()
            objgaspea.mutation()
            
        help_perem = objgansga.matrix_genotype
        objgansga.matrix_genotype = objgaspea.matrix_genotype
        objgaspea.matrix_genotype = help_perem
            
    all_individ_ph = objgansga.matrix_phenotype + objgaspea.matrix_phenotype
    
    massiv_one = [objgansga.matrix_phenotype[i] for i in range(number_half_individ) if objgansga.matrix_rank[i] == 1]
    help_vector = objgaspea.calcul_raw_fitness(func_all)
    massiv_two = [objgaspea.matrix_phenotype[i] for i in range(number_half_individ) if help_vector[i] == 0]
    massiv_all = massiv_one + massiv_two
       
    GD[iter] = generational_distance(func_all, all_individ_ph, massiv_all, objgansga.number_criteria)
    S[iter] = spasing(func_all, all_individ_ph, objgansga.number_criteria)
        
print(sum(GD) / number_iteration)
print(sum(S) / number_iteration)