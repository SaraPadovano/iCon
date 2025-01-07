import pandas as pd
from pyswip import Prolog

# Funzione per scrivere fatti riguardanti le case produttrici nella kb.pl dal dataset
def write_creators(dataset):
    df = pd.read_csv(dataset)

    # Otteniamo una lista unica della colonna creator
    creators = df['creator'].dropna().unique()

    with open('kb.pl', mode='w', encoding='utf-8') as kb:
        # Scrive le case produttrici
        for creator in creators:
            if creator:  # Con questo if ci assicuriamo la presenza del creator (controllo in più)
                kb.write(f"creator('{creator}').\n")

        kb.write('\n')

#Funzione che scrive un assioma nel file kb.pl controllando che non sia già presente
def write_fact_to_file(fact, kb):
    # Verifica se il fatto è già presente
    with open(kb, 'r', encoding='utf-8') as file:
        existing_content = file.read()

    if fact not in existing_content:
        # Riapri il file in modalità append e scrivi il fatto
        with open(kb, 'a', encoding='utf-8') as file:
            file.write(f"{fact}.\n")

# Funzione che scrive i fatti riguardanti le macchine nel kb file dal dataset richiamando write_fact_to_file
def write_auto_info(dataset, kb):
    with open(kb, "a", encoding="utf-8"):
        write_fact_to_file(":- encoding(utf8)", kb)
        file = pd.read_csv(dataset)
        for index, row in file.iterrows():
            name = repr(row['name'])
            mpg = int(row['mpg']) # Forzo a tutti gli interi il tipo perchè la libreria pandas mi stava dando problemi sul tipo di horsepower, nonostante fosse palesemente un intero
            cylinders = int(row['cylinders'])
            displacement = int(row['displacement'])
            horsepower = int(row['horsepower'])
            weight = int(row['weight'])
            acceleration = int(row['acceleration'])
            model_year = int(row['model_year'])
            creator = repr(row['creator'])
            price = int(row['price'])
            prolog_clause = f"auto({name},{mpg},{cylinders},{displacement},{horsepower},{weight},{acceleration},{model_year},{creator},{price})"
            write_fact_to_file(prolog_clause, kb)

# Al momento non sono presenti delle specifiche regole legate alla kb poichè non vi sono relazioni esplicite
# tra i vari dati che richiedono la formazione di regole per il sistema. 

# Funzione per fare query in Prolog
def execute_query(query, kb):
    prolog = Prolog()
    prolog.consult(kb)
    result = list(prolog.query(query))
    return result