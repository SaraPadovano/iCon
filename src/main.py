import pandas as pd
from auto_prolog import write_creators, write_auto_info
from features import recent

# Percorso dei file
fileName = "../dataset/Automobile.csv"
fileName_cleaned = "../dataset/Automobile_cleaned.csv"
fileName_features = "../dataset/Automobile_features.csv"
file_know_base = "kb.pl"

# pulizia del file da eventuali dati mancanti
try:
    df = pd.read_csv(fileName, on_bad_lines='skip')

    # Elimina le righe con valori mancanti dal dataset
    df_cleaned = df.dropna()

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
    df_cleaned.to_csv(fileName_features, index=False)
    df_features = pd.read_csv(fileName_features, on_bad_lines='skip')

    # Aggiungiamo la colonna recent al file ripulito
    recent(df_features)

except FileNotFoundError:
    print(f"Error: The file {fileName} was not found.")
except pd.errors.EmptyDataError:
    print(f"Error: The file {fileName} is empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


# RAGIONAMENTO LOGICO
write_creators(fileName_cleaned)
write_auto_info(fileName_cleaned, file_know_base)


