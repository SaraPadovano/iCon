from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, K2Score
import pickle

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
    bn.fit(dataset, estimator=MaximumLikelihoodEstimator, n_jobs=2)

    with open('../dataset/bayesian_network.pkl', 'wb') as output:
        pickle.dump(bn, output)

    return bn