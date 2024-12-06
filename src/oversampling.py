import pandas as pd
import numpy as np
import resreg
def oversampling_smoter(X, y, target, relevance):
    # Controllo che le colonne siano numeriche (per sicurezza)
    X = pd.DataFrame(X)
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    relevance = pd.to_numeric(relevance, errors='coerce')

    # Controlla se ci sono valori NaN nel dataset e rimuovili (se necessario)
    if X.isnull().values.any():
        X = X.fillna(0)

    # Chiamo la funzione smoter da resreg
    train_data_resample = resreg.smoter(X, y, relevance=relevance, random_state=42)

    #Estraiamo X e y dal nuovo dataset
    X = train_data_resample.drop(columns=[target]).to_numpy()
    y = train_data_resample[target].to_numpy()

    dataset_resampled = pd.DataFrame(X, columns=X.columns)
    dataset_resampled[target] = y

    # Salvo il dataframe in un file CSV
    file_path = "../dataset/Auto_resampled.csv"
    dataset_resampled.to_csv(file_path, index=False)  # Salva senza l'indice nel file CSV

    print(f"File salvato in {file_path}")

    return X, y