import pandas as pd
from auto_prolog import write_creators, write_auto_info
from features import bar_distribution, recent, convert_true_false, normalize_price, normalize_integer
from unsupervised_learning import cluster
from supervised_learning import train_valuate_model
from oversampling import oversampling_smogn
from sklearn.preprocessing import KBinsDiscretizer
from bayesian_net import create_bayesian_network, visualize_bayesian_network, show_cpd, generate_random_example
from pgmpy.inference import VariableElimination
import numpy as np

# Percorso dei file
fileName = "../dataset/Automobile.csv"
fileName_cleaned = "../dataset/Automobile_cleaned.csv"
fileName_features = "../dataset/Automobile_features.csv"
fileName_clusters = "../dataset/Automobile_clusters.csv"
fileName_resampled = "../dataset/Automobile_resampled.csv"
file_know_base = "kb.pl"

# pulizia del file da eventuali dati mancanti
try:
    df = pd.read_csv(fileName, on_bad_lines='skip')

    # Elimina le righe con valori mancanti dal dataset
    df_cleaned = df.dropna()

    # Evita il downcast silenzioso
    pd.set_option('future.no_silent_downcasting', True)

    # Controlla che le colonne numeriche siano effettivamente numeriche e eventualmente le converte
    numeric_columns = [
        'mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'price'
    ]
    for column in numeric_columns:
        df_cleaned[column] = pd.to_numeric(df[column], errors='coerce')

    # Salva il file pulito in un nuovo file
    df_cleaned.to_csv(fileName_cleaned, index=False)
    print(f"File saved successfully to {fileName_cleaned}")

    # Salvo il file completamente ripulito in un altro file per il feature engineering
    df_features = pd.read_csv(fileName_cleaned, encoding='utf-8-sig')

    recent(df_features)
    creators = [
        'chevrolet', 'buick', 'plymouth', 'amc', 'ford', 'pontiac', 'dodge', 'toyota', 'datsun',
        'peugeot', 'audi', 'saab', 'bmw', 'opel', 'fiat', 'volkswagen', 'mercury', 'oldsmobile',
        'chrysler', 'mazda', 'volvo', 'renault', 'honda', 'mercedes', 'subaru', 'nissan', 'porsche',
        'ferrari', 'mitsubishi', 'jeep', 'jaguar', 'lamborghini'
    ]
    df_features = pd.get_dummies(df_features, columns=['creator'], prefix='', prefix_sep='')
    df_features = convert_true_false(df_features)
    normalize_price(df_features)
    normalize_integer(df_features)
    features = [
                   'model_year', 'recent', 'normalized_mpg', 'cylinders',
                   'normalized_displacement', 'normalized_horsepower',
                   'normalized_weight', 'normalized_acceleration'
               ] + creators
    final_df = df_features[features + ['log_price']]
    # Salva il dataset finale in un nuovo file
    final_df.to_csv(fileName_features, index=False)
    # Crea un grafo che mostra la distribuzione dei creators
    creator_distribution = df_features[creators].sum().sort_values()
    bar_distribution(creator_distribution, 'Creator Distribution', 'creator', 'occurences',
                     '../png/creator_distribution.png')


except FileNotFoundError:
    print(f"Error: The file {fileName} was not found.")
except pd.errors.EmptyDataError:
    print(f"Error: The file {fileName} is empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# RAGIONAMENTO LOGICO
write_creators(fileName_cleaned)
write_auto_info(fileName_cleaned, file_know_base)

# APPRENDIMENTO NON SUPERVISIONATO
print("Inizio apprendimento non supervisionato")
df = pd.read_csv(fileName_features, encoding='utf-8-sig')
features = [
    'recent', 'normalized_mpg', 'cylinders', 'normalized_displacement', 'normalized_horsepower',
    'normalized_weight', 'normalized_acceleration', 'chevrolet', 'buick', 'plymouth', 'amc', 'ford', 'pontiac',
    'dodge', 'toyota', 'datsun', 'peugeot', 'audi', 'saab', 'bmw', 'opel', 'fiat', 'volkswagen', 'mercury',
    'oldsmobile', 'chrysler', 'mazda', 'volvo', 'renault', 'honda', 'mercedes', 'subaru', 'nissan', 'porsche',
    'ferrari', 'mitsubishi', 'jeep', 'jaguar', 'lamborghini'
]
clusters, centroids = cluster(df, features,'../png/best_k', '../png/distribution_cars_in_clusters', fileName_clusters)
print("Fine apprendimento non supervisionato")

# APPRENDIMENTO SUPERVISIONATO
#print("Inizio apprendimento supervisionato")
#df = pd.read_csv(fileName_clusters, encoding='utf-8-sig')
# Assicura che la colonna target sia di tipo numerico
#targetColumn = 'log_price'
#df[targetColumn] = pd.to_numeric(df[targetColumn], errors='coerce')
#df_copy = df.copy()
#X = df_copy.drop(columns=[targetColumn]).to_numpy()
#y = df_copy[targetColumn].to_numpy()
#model = train_valuate_model(X, y, o=False)
#print("Fine apprendimento supervisionato")


# OVERSAMPLING
#df = pd.read_csv(fileName_clusters, encoding='utf-8-sig')
#targetColumn = 'log_price'
#df[targetColumn] = pd.to_numeric(df[targetColumn], errors='coerce')
#dfOver = df.copy()
#dfOver = dfOver.replace({True: 1, False: 0}).infer_objects(copy=False)
#X = dfOver.drop(columns=[targetColumn]).to_numpy()
#y = dfOver[targetColumn].to_numpy()
#assert len(X) == len(y)
#print("Inizio oversampling")
#X_over, y_over = oversampling_smogn(X, y, targetColumn)
#print("Fine oversampling")
# addestriamo i modelli dopo l'oversampling
#print("Addestriamo i modelli dell'apprendimento supervisionato dopo l'oversampling")
#model_oversampled = train_valuate_model(X_over, y_over, o=True)

# APPRENDIMENTO PROBABILISTICO
print("Inizio apprendimento probabilistico")
df_prob = pd.read_csv('../dataset/Automobile_cleaned.csv', encoding='utf-8-sig')
# Assicuriamoci che la feature categorica creator sia trattata come tale
categorical_column = 'creator'
df_prob[categorical_column] = df_prob[categorical_column].astype('category')
# Discretizziamo le variabili continue
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
df_prob['mpg'] = discretizer.fit_transform(df_prob[['mpg']])
df_prob['displacement'] = discretizer.fit_transform(df_prob[['displacement']])
df_prob['horsepower'] = discretizer.fit_transform(df_prob[['horsepower']])
df_prob['weight'] = discretizer.fit_transform(df_prob[['weight']])
df_prob['acceleration'] = discretizer.fit_transform(df_prob[['acceleration']])
df_prob['price'] = discretizer.fit_transform(df_prob[['price']])
# Eliminiamo la colonna name che non ci serve
df_prob.drop(columns=['name'], axis=1, inplace=True)
# Creo adesso la mia rete bayesiana
print("Creazione della rete bayesiana")
bn = create_bayesian_network(df_prob)
# Visualizzo la rete bayesiana
print("Visualizzazione della rete")
visualize_bayesian_network(bn)
print("Calcoliamo il cpd delle variabili")
show_cpd(bn)
print("Generiamo un esempio randomico")
random_example = generate_random_example(bn)
print("Example: " + str(random_example))
inference = VariableElimination(bn)
result = inference.query(variables=['price'], evidence=random_example.iloc[0].to_dict())
print(result)
print("Fine apprendimento probabilistico")

