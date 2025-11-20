import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Definir categorias e parâmetros
categorias = ['Eletronicos', 'Vestuário', 'Alimentos', 'Brinquedos', 'Livros']
n = 100  # número de registros

# Gerar dados aleatórios
np.random.seed(42)  # para reprodutibilidade

dados = {
    'produto_id': np.arange(1, n + 1),
    'categoria': np.random.choice(categorias, n),
    'preco': np.round(np.random.uniform(5, 2000, n), 2),
    'quantidade': np.random.randint(1, 20, n),
    'data_venda': [
        (datetime(2024, 1, 1) + timedelta(days=int(x))).strftime("%Y-%m-%d")
        for x in np.random.randint(0, 365, n)
    ]
}

# Criar DataFrame
df = pd.DataFrame(dados)

# Salvar em CSV
df.to_csv("vendas.csv", index=False)

print("Arquivo 'vendas.csv' criado com sucesso!")
print(df.head())
