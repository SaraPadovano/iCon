from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, K2Score
import pickle
import networkx as nx
from matplotlib import pyplot as plt

# Funzione che crea la rete bayesiana
def create_bayesian_network(dataset):
    edges = []
    edges.append(('price', 'model_year'))
    edges.append(('price', 'creator'))
    edges.append(('price', 'cylinders'))
    edges.append(('cylinders', 'mpg'))
    edges.append(('cylinders', 'displacement'))
    edges.append(('cylinders', 'horsepower'))
    edges.append(('displacement', 'weight'))
    edges.append(('displacement', 'horsepower'))
    edges.append(('horsepower', 'acceleration'))
    edges.append(('horsepower', 'weight'))
    edges.append(('horsepower', 'mpg'))
    edges.append(('weight', 'mpg'))
    edges.append(('weight', 'acceleration'))

    bn = BayesianNetwork(edges)
    bn.fit(dataset, estimator=MaximumLikelihoodEstimator, n_jobs=-1)
    # Salvo la rete bayesiana su file
    with open('../dataset/Bayesian_network.pkl', 'wb') as output:
        pickle.dump(bn, output)
    return bn

# Funzione che visualizza la rete bayesiana
def visualize_bayesian_network(bayesian_network: BayesianNetwork):
    G = nx.MultiDiGraph(bayesian_network.edges())
    pos = nx.spring_layout(G, iterations=100, k=2, threshold=5, pos=nx.spiral_layout(G))
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=700,
        node_color="#6fa8dc",  # Colore blu tenue
        edgecolors="#2c3e50",  # Bordo scuro
        linewidths=2
    )
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=12,
        font_color="#2c3e50",
        font_weight="bold",
        clip_on=True,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=15,
        arrowstyle="-|>",
        edge_color="#e74c3c",
        connectionstyle="arc3,rad=0.2",
        min_source_margin=1.2,
        min_target_margin=1.5,
        edge_vmin=2,
        edge_vmax=2,
    )
    plt.title("Bayesian Network Graph", fontsize=16, fontweight="bold", color="#34495e")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig('../png/Bayesian_network_graph.png')
    plt.clf()

# Carica la rete bayesiana
def load_bayesian_network():
    with open('../dataset/Bayesian_network.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Calcolo le distribuzione di prababilità condizionata
def show_cpd(bayesian_network: BayesianNetwork):
    cpd_list = bayesian_network.get_cpds()
    with open('../text/bayesian_network_cpd.txt', 'w') as file:
        for cpd in cpd_list:
            file.write(f'CPD of {cpd.variable}:\n')
            file.write(str(cpd) + '\n\n')