import pandas as pd
import numpy as np

from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN, FILE_OUT
from pycompss.api.api import compss_wait_on


# ------------------------------------------------------------
# TAREFA PARA PROCESSAR UMA CATEGORIA
# ------------------------------------------------------------
@task(data_chunk=FILE_IN, output_file=FILE_OUT)
def processar_categoria(data_chunk, output_file):
    """Processa estatísticas de uma categoria e salva CSV."""
    df = pd.read_csv(data_chunk)

    media_preco = df["preco"].mean()
    desvio_preco = df["preco"].std()
    total_unidades = df["quantidade"].sum()
    receita_total = (df["preco"] * df["quantidade"]).sum()

    resultado = pd.DataFrame({
        "categoria": [df["categoria"].iloc[0]],
        "media_preco": [media_preco],
        "desvio_preco": [desvio_preco],
        "total_unidades": [total_unidades],
        "receita_total": [receita_total]
    })

    resultado.to_csv(output_file, index=False)


# ------------------------------------------------------------
# FUNÇÃO PRINCIPAL DO WORKFLOW
# ------------------------------------------------------------
def workflow_vendas(input_csv="vendas.csv"):
    # Lê o dataset completo (execução serial aqui)
    df = pd.read_csv(input_csv)

    # Agrupa por categoria
    categorias = df["categoria"].unique()

    tarefas = []

    for cat in categorias:
        df_cat = df[df["categoria"] == cat]

        # Salva temporariamente o grupo em CSV para o task
        temp_file = f"tmp_categoria_{cat}.csv"
        df_cat.to_csv(temp_file, index=False)

        # Nome do arquivo de saída
        output_file = f"resultado_{cat}.csv"

        # Lança tarefa PyCOMPSs para processar a categoria
        t = processar_categoria(temp_file, output_file)
        tarefas.append(t)

    # Espera todas as tarefas terminarem
    compss_wait_on(tarefas)

    print("Processamento concluído. Arquivos gerados por categoria.")


# ------------------------------------------------------------
# EXECUÇÃO DIRETA
# ------------------------------------------------------------
if __name__ == "__main__":
    workflow_vendas()
