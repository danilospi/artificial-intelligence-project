import random
import math
import numpy as np

class Graph:
    def __init__(self):
        self.vertices = []
        self.edges = []
        self.matrix = np.array([0, 0])

    def __str__(self):
        return 'Nodes: {}, Edges: {}.'.format(len(self.vertices), len(self.edges))

    def selectConnectedVertexByVertex(self, vertex):
        vertex_list = []
        for i in range(len(self.matrix[vertex-1])):
            if(self.matrix[vertex-1][i] == 1):
                vertex_list.append(i+1)
        return vertex_list


class EquitableGraphColorizer:

    def __init__(self, filename, parameters):
        self.graph = self.loadGraphGraph(filename)
        self.iteration_number = parameters['run_number']
        self.max_generation = parameters['max_generation']
        self.population_size = parameters['population_size']
        self.mutation_probability = parameters['mutation_probability']
        self.crossover_probability = parameters['crossover_probability']
        self.max_improvements = parameters['max_improvements']
        self.random_seed = parameters['random_seed']
        self.size_vertex = len(self.graph.vertices)
        self.size_edge = len(self.graph.edges)
        self.k_best = 100000
        self.improvement = 0
        self.run_i = 0

    def findKMin(self):
        results = []
        color_size = []
        for n in range(self.iteration_number):
            print("RUN NUMBER: ", n+1)
            self.improvement = 0
            self.k_best = 100000
            ris = self.applyGeneticAlgorithm()
            if ris:
                best = min(ris, key=lambda tup: tup[1])
                results.append(best[0])
                color_size.append(best[1])
            self.run_i += 1

        return results, color_size

    def applyGeneticAlgorithm(self):
        """
        Metodo per eseguire tutti i passi dell'algoritmo genetico:
        1) Inizializzazione della popolazione
        2) Selezione degli individui migliori
        3) Operazione di crossover
        4) Operazione di mutazione
        """
        solutions = []
        population = self.initializePopulation()
        t = 0
        while not self.stopConditionReached(t, solutions):
            if(t % 1000 == 0):
                print("GENERATION: ", t, " IN: ", self.max_generation)
            population = self.selectEvenPopulation(population)
            population = self.applyMergeCrossover(population)
            population = self.applyMutationShift(population)
            solutions.extend(self.selectAmmissibleSolution(population))
            t += 1
            self.improvement += 1
        return solutions

    def initializePopulation(self):
        """
        Metodo per inizializzare la popolazione.
        Ad ogni vertice del grafo viene assegnato un colore random preso dalla lista
        'colorList' e poi il colore viene rimosso dalla lista. Non appena la lista risulta vuota
        i colori vengono assegnati randomicamente. In questo modo ottengo un grafo colororato in maniera
        quasi equa.
        Questo metodo restituisce una lista di elementi di questo tipo:
        [(1,3), (2,0), (3,1), (4,3),....]
        il primo numero corrisponde al vertice, mentre il secondo corrisponde al colore.
        """
        v = self.graph.vertices
        print(self.random_seed[self.run_i])
        random.seed(self.random_seed[self.run_i])
        population = []
        while len(population) < self.population_size:
            colors = [i for i in range(len(v))]
            coloring = []
            for i in range(len(v)):
                vertex = i + 1
                chosen_color = random.choice(colors)
                coloring.append((vertex, chosen_color))
            x = self.fixColorNumber(coloring)
            population.append(x)
        return population


    def calculateFitnessFunction(self, individual):
        e = self.graph.edges
        v = self.graph.vertices
        c = individual
        index = 1
        cardinality = []
        conflictPenaltyFunction = 0
        equityPenaltyFunction = 0
        for w1, w2 in e:
            if(self.getColor(w1, c) == self.getColor(w2, c)):
                conflictPenaltyFunction += 1

        for vertex in v:
            color = self.getColor(vertex, c)
            addColor = True
            for clr, num in cardinality:
                if clr == color:
                    i = cardinality.index((clr,num))
                    num += 1
                    cardinality[i] = (clr, num)
                    addColor = False

            if(addColor == True):
                cardinality.append((color, index))

        for clr, num in cardinality:
            theoreticalCardinalityP = math.ceil(len(self.graph.vertices) / len(cardinality))
            theoreticalCardinalityN = math.floor(len(self.graph.vertices) / len(cardinality))
            if (num >= theoreticalCardinalityP):
                equityPenaltyFunction += num - theoreticalCardinalityP
            elif (num <= theoreticalCardinalityN):
                equityPenaltyFunction += theoreticalCardinalityN - num

        return equityPenaltyFunction + conflictPenaltyFunction, len(cardinality)



    def getColor(self, vertex, coloring):
        """
        Metodo per trovare il colore di un vertice.
        """
        for v, c in coloring:
            if v == vertex:
                return c

    def selectEvenPopulation(self, population):
        #random.seed(self.random_seed[self.run_i])
        after_selection = []
        list_tmp = []
        for individual in population:
            fit1, fit2 = self.calculateFitnessFunction(individual)
            list_tmp.append((individual, fit1+fit2))

        for i in range(len(list_tmp)):
            individual1 = random.choice(list_tmp)
            individual2 = random.choice(list_tmp)
            if(individual1[1] < individual2[1]):
                after_selection.append(individual1[0])
            else:
                after_selection.append(individual2[0])

        return after_selection


    def selectAmmissibleSolution(self, allPopulation):
        """
        Metodo per controllare se esistono soluzioni ammissibili
        nella popolazione.
        """
        sol = []
        for individual in allPopulation:
            fit1, fit2 = self.calculateFitnessFunction(individual)
            if(fit1 == 0):
                sol.append((individual, fit2))
                if self.k_best > fit2:
                    self.k_best = fit2
                    self.improvement = 0
                    print("FIND NEW BEST")
        return sol

    def calculateCardinality(self, individual):
        """
        Metodo per calcolare il numero di vertici che fanno parte della
        stessa classe di colore.
        Questo metodo restituisce qualcosa del tipo:
        [(0,5), (1,2), (2,7), (3,1),.....]
        dove il primo elemento rapprensenta il colore, mentre il secondo
        rappresenta il numero di elementi presenti in quella classe di colore
        """
        v = self.graph.vertices
        c = individual
        index = 1
        cardinality = []
        for vertex in v:
            color = self.getColor(vertex, c)
            addColor = True
            for clr, num in cardinality:
                if clr == color:
                    i = cardinality.index((clr,num))
                    num += 1
                    cardinality[i] = (clr, num)
                    addColor = False

            if(addColor == True):
                cardinality.append((color, index))
        cardinality = sorted(cardinality, key=lambda tup: tup[0])
        return cardinality

    def stopConditionReached(self, t, solutions):
        """
        Metodo per verificare se proseguire con l'esecuzione dell'algoritmo
        genetico, o se fermare il tutto.
        """
        if t > self.max_generation:
            return True

        elif self.improvement > self.max_improvements:
            return True

        return False

    def applyMutationShift(self, p_before_mutation):
        #random.seed(self.random_seed[self.run_i])
        p_after_mutation = []

        for individual in p_before_mutation:
            if random.uniform(0.0, 1.0) < self.mutation_probability:
                cardinality = self.calculateCardinality(individual)
                minimum = min(cardinality, key = lambda t: t[1])
                index = random.randint(0,len(self.graph.vertices)-1)
                individual[index] = (individual[index][0], minimum[0])
            p_after_mutation.append(self.fixColorNumber(individual))

        return p_after_mutation

    def fixColorNumber(self, coloring):
        sort_by_color = sorted(coloring, key=lambda tup: tup[1])
        i = 0
        tmp = sort_by_color[0][1]
        individual = []
        for vtx, col in sort_by_color:
            if (tmp != col):
                i += 1
                tmp = col
            individual.append((vtx, i))
        individual = sorted(individual, key=lambda tup: tup[0])
        return individual

    def loadGraphGraph(self, path):
        """
        Questo metodo prende in input un file di tipo '.col' e lo
        converte in un grafo
        """
        graph = Graph()
        vertexI = 0

        with open(path, mode='r') as f:
            num_vertex, num_edge = f.readline().split()
            ad_matrix = np.zeros([int(num_vertex), int(num_vertex)], dtype=int)
            for line in f.readlines():

                vtx_list = line.split()
                for vtx in vtx_list:
                    ad_matrix[int(vertexI)][int(vtx)-1] = 1
                vertexI += 1

            for vertex in range(int(num_vertex)):
                graph.vertices.append(vertex+1)

            copy = np.copy(ad_matrix)
            edge_list = []
            np.fill_diagonal(copy, -1)
            for i in range(int(num_vertex)):
                for j in range(int(num_vertex)):
                    if(copy[i][j] == -1):
                        break
                    if(copy[i][j] == 1):
                        edge_list.append((i+1, j+1))
                        graph.edges.append((i+1, j+1))

            graph.matrix = ad_matrix

        return graph


    def applyMergeCrossover(self, p_before_crossover):
        #random.seed(self.random_seed[self.run_i])
        p_after_cross = []
        size = len(self.graph.vertices)
        for i in range(int(len(p_before_crossover))):
            individual1 = random.choice(p_before_crossover)
            individual2 = random.choice(p_before_crossover)

            merge1 = []
            merge2 = []
            oldIndex = 0
            i = 0
            while len(merge1) < size:
                index = random.randint(0, int(size/2))
                if i%2 == 0:
                    if (oldIndex + index < size):
                        merge1.extend(individual1[oldIndex:index+oldIndex])
                        merge2.extend(individual2[oldIndex:index+oldIndex])
                    else:
                        merge1.extend(individual1[oldIndex:size])
                        merge2.extend(individual2[oldIndex:size])
                else:
                    if(oldIndex+index < size):
                        merge2.extend(individual1[oldIndex:index+oldIndex])
                        merge1.extend(individual2[oldIndex:index+oldIndex])
                    else:
                        merge2.extend(individual1[oldIndex:size])
                        merge1.extend(individual2[oldIndex:size])

                oldIndex += index
                i += 1

            merge1 = self.fixColorNumber(merge1)
            merge2 = self.fixColorNumber(merge2)
            fit1 = self.colorSize(merge1)
            fit2 = self.colorSize(merge2)

            if(fit1 < fit2):
                p_after_cross.append(merge1)
            else:
                p_after_cross.append(merge2)

        return p_after_cross

    def colorSize(self, individual):
        sort_by_color = sorted(individual, key=lambda tup: tup[1])
        size = sort_by_color[len(sort_by_color)-1][1]
        return size
