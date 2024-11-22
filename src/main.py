import pandas as pd
from auto_prolog import write_creators, write_auto_info

# Percorso dei file
fileName = "../dataset/Automobile.csv"
fileName_cleaned = "../dataset/Automobile_cleaned.csv"
file_know_base = "kb.pl"

# pulizia del file da eventuali dati mancanti
try:
    df = pd.read_csv(fileName, on_bad_lines='skip')

    # Elimina le righe con valori mancanti dal dataset
    df_cleaned = df.dropna()

    # Salva il file pulito in un nuovo file
    df_cleaned.to_csv(fileName_cleaned, index=False)
    print(f"File saved successfully to {fileName_cleaned}")

except FileNotFoundError:
    print(f"Error: The file {fileName} was not found.")
except pd.errors.EmptyDataError:
    print(f"Error: The file {fileName} is empty.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


#RAGIONAMENTO LOGICO
write_creators(fileName_cleaned)
write_auto_info(fileName_cleaned, file_know_base)


