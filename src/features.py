import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Data una feature realizza un grafico che verrà salvato come immagine
def bar_distribution(data, title, xlabel, ylabel, file):
    # Imposta lo stile
    plt.style.use('ggplot')
    # Imposta la grandezza dell'immagine
    plt.figure(figsize=(8, 6))
    # Crea un grafico a barre
    ax = data.plot(kind='bar', color='#4CAF50', edgecolor='#2C6B2F', width=0.8)
    # Imposta titolo ed etichette degli assi
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    # Aggiunge una griglia per migliorare la leggibilità
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    # Salva l'immagine
    plt.tight_layout()  # Ottimizza il layout
    plt.savefig(file)

# Funzione per la creazione della colonna Recent
def recent(file):
    file['recent'] = file['model_year'] >= 2015

# Funzione che converte True e False in 1 e 0
def convert_true_false(file):
    file = file.replace({True: 1, False: 0}).infer_objects(copy=False)

# Funzione per normalizzare price con il log1 (per gestire eventuali 0)
def normalize_price(file):
    file['log_price'] = np.log1p(file['price'])

#Funzione per normalizzare tutti i valori interi in base alla tecnica più appropriata
def normalize_integer(file):
    # Standardizzazione per l'mpg, il displacement, l'horsepower, il weight e l'acceleration
    scaler = StandardScaler()
    file['standardized_mpg'] = scaler.fit_transform(file[['mpg']])
    file['standardized_displacement'] = scaler.fit_transform(file[['displacement']])
    file['standardized_horsepower'] = scaler.fit_transform(file[['horsepower']])
    file['standardized_weight'] = scaler.fit_transform(file[['weight']])
    file['standardized_acceleration'] = scaler.fit_transform(file[['acceleration']])
    # Per cylinders nessuna normalizzazione perchè i valori sono limitati a solitamente 4,6,8 e raramente 2
    # una normalizzazione quindi non aggiungerebbe valore


