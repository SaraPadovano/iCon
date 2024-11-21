import pandas as pd
# from pyswip import Prolog

# Funzione per scrivere fatti riguardanti le case produttrici nella kb.pl dal dataset

def write_creators(dataset_path):
    df = pd.read_csv(dataset_path)

    # Otteniamo una lista unica della colonna creator
    creators = df['creator'].dropna().unique()

    with open('kb.pl', mode='w', encoding='utf-8') as kb:
        # Scrive le case produttrici
        for creator in creators:
            if creator:  # Con questo if ci assicuriamo la presenza del creator (controllo in più)
                kb.write(f"creator('{creator}').\n")

        kb.write('\n')
