import pandas as pd
from auto_prolog import write_creators, write_auto_info
from features import bar_distribution, recent, convert_true_false, normalize_price, normalize_integer

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
    convert_true_false(df_features)
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
