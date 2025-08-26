import pandas as pd
import random
import json
import constants

def prepare():
    # Carregar o dataset original
    df = pd.read_csv(constants.DATASET_PATH, header=None, names=['ceares', 'portugues'])

    # Criar dicionário de mapeamento
    ceares_dict = {}
    for _, row in df.iterrows():
        portugues_terms = [term.strip() for term in row['portugues'].lower().split(';')]
        for term in portugues_terms:
            ceares_dict[term] = row['ceares'].lower()

    # Gerar pares de treinamento
    train_pairs = []

    # Padrões de frases para treinamento
    patterns = [
        "Isso é {term}",
        "Ele é {term}",
        "Ela está {term}",
        "Você é {term}",
        "Isso parece {term}",
        "Aquilo foi {term}",
        "Eu sou {term}",
        "Nós estamos {term}",
        "O {term} é bom",
        "A {term} é interessante"
    ]

    for pt_term, ce_term in ceares_dict.items():
        for pattern in patterns:
            if random.random() < 0.8:  # 80% para treino
                train_pairs.append({
                    'original': pattern.format(term=pt_term),
                    'ceares': pattern.format(term=ce_term)
                })

    # Salvar dados de treinamento
    with open(constants.TRAIN_DATA_PATH, 'w', encoding='utf-8') as file:
        json.dump(train_pairs, file, ensure_ascii=False, indent=2)

    print(f"Gerados {len(train_pairs)} pares de treinamento em {constants.TRAIN_DATA_PATH}")
