import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Data una feature realizza un grafico che verrà salvato come immagine
def bar_distribution(data, title, xlabel, ylabel, file):
    # Imposta lo stile
    plt.style.use('seaborn-bright')
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