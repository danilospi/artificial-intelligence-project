from equitable_graph_coloring import EquitableGraphColorizer
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import *

if __name__ == '__main__':
    path = 'testGraph/queen6_6.graph'
    params = {
        'run_number': 5,
        'max_generation': 1000,
        'population_size': 20,
        'mutation_probability': 0.8,
        'crossover_probability': 0.8,
        'max_improvements': 800,
        'random_seed': [1936,6029,3573,4454,9163]
    }

    egc = EquitableGraphColorizer(path, params)
    result, color_size = egc.findKMin()
    mySolution = []
    k_best = 0
    m = 0
    dev_std = 0

    if(result):
        k_best = min(color_size)
        m = np.mean(color_size)
        dev_std = np.std(color_size)
        print("K_BEST: ", k_best)
        print("K_MEAN: ", m)
        print("K_STD: ", dev_std)

    if result:
        mySolution = result[color_size.index(k_best)]
    all_color = []
    all_color_number = []
    colors = []

    for y, u in mySolution:
        colors.append(u)

    color_class = egc.calculateCardinality(mySolution)
    for i in range(len(color_class)):
        all_color.append(i)
        all_color_number.append(color_class[i][1])

    print("SOLUTION: ",mySolution)

    G = nx.Graph()
    for x in egc.graph.vertices:
        G.add_node((x))
    for i in egc.graph.edges:
        G.add_edge(i[0], i[1])

    nx.draw(G, with_labels=True, node_color=colors, label=all_color_number)
    print("K: ",len(all_color))
    color_patch = mpatches.Patch(label=all_color_number)
    plt.legend(handles=[color_patch])
    plt.show()