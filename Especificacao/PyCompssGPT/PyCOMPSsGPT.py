# ============================================
# Workflow PyCOMPSs - Análise de Vendas por Categoria
# ============================================

import pandas as pd
import numpy as np
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on, compss_barrier
import os

# -------------------------------------------------
# Task: Calcular métricas por categoria
# -------------------------------------------------
@task(category=str, df=pd.DataFrame, returns=str)
def process_category(category, df):
    """
    Calcula métricas agregadas para uma categoria de produtos.
    Retorna o nome do arquivo CSV salvo.
    """
    mean_price = df['preco'].mean()
    std_price = df['preco'].std()
    total_units = df['quantidade'].sum()
    total_revenue = (df['preco'] * df['quantidade']).sum()

    # Cria DataFrame com o resumo
    summary = pd.DataFrame({
        'categoria': [category],
        'media_preco': [mean_price],
        'desvio_padrao_preco': [std_price],
        'total_unidades': [total_units],
        'receita_total': [total_revenue]
    })

    # Salva o resultado como CSV
    filename = f"saida_{category}.csv"
    summary.to_csv(filename, index=False)
    return filename

# -------------------------------------------------
# Função principal
# -------------------------------------------------
def main():
    # Caminho para o dataset de entrada
    input_file = "vendas.csv"

    # Leitura dos dados
    df = pd.read_csv(input_file)

    # Agrupa por categoria
    grouped = df.groupby('categoria')

    # Lista para armazenar as tasks
    output_files = []

    # Cria uma task por categoria
    for category, group in grouped:
        output = process_category(category, group)
        output_files.append(output)

    # Espera todas as tasks terminarem
    output_files = compss_wait_on(output_files)

    print("\nArquivos gerados:")
    for f in output_files:
        print(f"- {f}")

    compss_barrier()
    print("\nWorkflow finalizado com sucesso.")

# -------------------------------------------------
# Execução padrão (fora do COMPSs, para testes)
# -------------------------------------------------
if __name__ == "__main__":
    main()
