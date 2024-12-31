from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import pickle
import networkx as nx
from matplotlib import pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd

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

# Funzione che genera l'esempio randomico
def generate_random_example(bn: BayesianNetwork):
    sample = bn.simulate(n_samples=1).drop(columns=['price'], axis=1)
    return sample

# Funzione per controllare che i valori inseriti siano numeri int o float
def is_number(val):
    try:
        float(val)  # Prova a convertire in float
        return True
    except ValueError:
        return False

# Funzione per la lettura da tastiera dei valori dell'esempio
def user_example_generate(bn: BayesianNetwork, discretizer_user):
    try:
        mpg = input("Inserire l'mpg (valore minimo 5 e massimo 200):")
        mpg = float(mpg)
        while mpg<5.0 or mpg>200.0 or is_number(mpg) is False or mpg == "":
            mpg = input("Inserire un valore valido:")
            mpg = float(mpg)
        if mpg.is_integer():
            mpg = int(mpg)
    except ValueError:
        print("Valore errato dell'mpg!")

    try:
        cylinders = int(input("Inserire il numero di cilindri (valori possibili: 2, 3, 4, 6, 8, 10, 12):"))
        valid_cylinders = {2, 3, 4, 6, 8, 10, 12}
        while cylinders not in valid_cylinders or cylinders == "":
            cylinders = int(input("Inserire un valore valido:"))
    except ValueError:
        print("Valore errato di cylinders!")

    try:
        displacement = input("Inserire la cilindrata (valore minimo 70 e massimo 500):")
        displacement = float(displacement)
        while displacement<70.0 or displacement>500.0 or is_number(displacement) is False or displacement == "":
            displacement = input("Inserire un valore valido:")
            displacement = float(displacement)
        if displacement.is_integer():
            displacement = int(displacement)
    except ValueError:
        print("Valore errato di displacement!")

    try:
        horsepower = input("Inserire la potenza (valore minimo 50 e massimo 400):")
        horsepower = float(horsepower)
        while horsepower<50.0 or horsepower>400.0 or is_number(horsepower) is False or horsepower == "":
            horsepower = input("Inserire un valore valido:")
            horsepower = float(horsepower)
        if horsepower.is_integer():
            horsepower = int(horsepower)
    except ValueError:
        print("Valore errato di horsepower!")

    try:
        weight = int(input("Inserire il peso (valore minimo 1000 e massimo 5000):"))
        while weight<1000 or weight>5000 or is_number(weight) is False or weight == "":
            weight = int(input("Inserire un valore valido:"))
    except ValueError:
        print("Valore errato di weight!")

    try:
        acceleration = input("Inserire l'accelerazione (valore minimo 2 e massimo 25):")
        acceleration = float(acceleration)
        while acceleration<2.0 or acceleration>25.0 or is_number(acceleration) is False or acceleration == "":
            acceleration = input("Inserire un valore valido:")
            acceleration = float(acceleration)
        if acceleration.is_integer():
            acceleration = int(acceleration)
    except ValueError:
        print("Valore errato di accelerazione!")

    try:
        model_year = int(input("Inserire l'anno del modello (valore minimo 1970 e massimo 2020):"))
        while model_year<1970 or model_year>2020 or is_number(model_year) is False or model_year == "":
            model_year = int(input("Inserire un valore valido:"))
    except ValueError:
        print("Valore errato dell'anno del modello!")

    try:
        creator = input("Inserire la caso automobilista di produzione del modello\n Scrivere una delle seguenti opzioni:\n"
                        "-chevrolet\n -buick\n -plymouth\n -amc\n -ford\n -pontiac\n "
                        "-dodge\n, -toyota\n -datsun\n -peugeot\n -audi\n -saab\n -bmw\n -opel\n -fiat\n -volkswagen\n "
                        "-mercury\n -oldsmobile\n -chrysler\n -mazda\n -volvo\n -renault\n -honda\n -mercedes\n -subaru\n "
                        "-nissan\n -porsche\n -ferrari\n -mitsubishi\n -jeep\n -jaguar\n -lamborghini\n")
        valid_creators = {'chevrolet', 'buick', 'plymouth', 'amc', 'ford', 'pontiac',
    'dodge', 'toyota', 'datsun', 'peugeot', 'audi', 'saab', 'bmw', 'opel', 'fiat', 'volkswagen', 'mercury',
    'oldsmobile', 'chrysler', 'mazda', 'volvo', 'renault', 'honda', 'mercedes', 'subaru', 'nissan', 'porsche',
    'ferrari', 'mitsubishi', 'jeep', 'jaguar', 'lamborghini'}
        while creator not in valid_creators or creator.isnumeric() is True or creator == "":
            creator = input("Inserire un valore valido:")
    except ValueError:
        print("Valore errato della casa automobilistica!")

    df_user = pd.DataFrame({
        'mpg': [mpg],
        'cylinders': [cylinders],
        'displacement': [displacement],
        'horsepower': [horsepower],
        'weight': [weight],
        'acceleration': [acceleration],
        'model_year': [model_year],
        'creator': [creator],
    })
    print(df_user)
    price = 0
    df_user_discretized = pd.DataFrame({
        'mpg': [mpg],
        'displacement': [displacement],
        'horsepower': [horsepower],
        'weight': [weight],
        'acceleration': [acceleration],
        'price': [price]
    })
    df_user_discretized = discretizer_user.transform(df_user_discretized)
    df_user_discretized = pd.DataFrame(df_user_discretized,
                                       columns=['mpg', 'displacement', 'horsepower', 'weight', 'acceleration', 'price'])

    user_input_discretized = {
        'mpg': df_user_discretized['mpg'].iloc[0],
        'cylinders': df_user['cylinders'][0],
        'displacement': df_user_discretized['displacement'].iloc[0],
        'horsepower': df_user_discretized['horsepower'].iloc[0],
        'weight': df_user_discretized['weight'].iloc[0],
        'acceleration': df_user_discretized['acceleration'].iloc[0],
        'model_year': df_user['model_year'][0],
        'creator': df_user['creator'][0]
    }
    print(user_input_discretized)

    sample = bn.simulate(n_samples=1).drop(columns=['price'], axis=1)

    # Sovrascriviamo i valori nel sample con i valori dell'utente
    for key, value in user_input_discretized.items():
        sample[key] = value
    print(sample)
    return sample