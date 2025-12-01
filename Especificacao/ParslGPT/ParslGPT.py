import parsl
from parsl import python_app, bash_app, DataFlowKernel, File
import pandas as pd
import os

# -------------------------------------------------------------------
# Configuração básica do Parsl (threads locais)
# -------------------------------------------------------------------
from parsl.config import Config
from parsl.executors.threads import ThreadPoolExecutor

parsl.load(
    Config(
        executors=[ThreadPoolExecutor(max_threads=4)],
        strategy=None
    )
)

# -------------------------------------------------------------------
# 1) App que processa cada categoria
# -------------------------------------------------------------------
@python_app
def processar_categoria(df_categoria, categoria, output_path):
    import pandas as pd

    preco_medio = df_categoria["preco"].mean()
    preco_std = df_categoria["preco"].std()
    total_unidades = df_categoria["quantidade"].sum()
    receita_total = (df_categoria["preco"] * df_categoria["quantidade"]).sum()

    resumo = pd.DataFrame({
        "categoria": [categoria],
        "preco_medio": [preco_medio],
        "preco_std": [preco_std],
        "total_unidades": [total_unidades],
        "receita_total": [receita_total]
    })

    resumo.to_csv(output_path, index=False)

    return f"Arquivo gerado: {output_path}"

# -------------------------------------------------------------------
# 2) Função principal do workflow
# -------------------------------------------------------------------
def workflow_vendas(input_csv, pasta_saida="resultados_parsl"):

    # Garante que a pasta de saída existe
    os.makedirs(pasta_saida, exist_ok=True)

    # Lê dataset
    df = pd.read_csv(input_csv)

    # Agrupa por categoria
    categorias = df["categoria"].unique()

    tarefas = []

    for cat in categorias:
        df_cat = df[df["categoria"] == cat]

        # Nome do arquivo de saída da categoria
        out_file = os.path.join(pasta_saida, f"{cat}_resumo.csv")

        # Executa tarefa paralela
        tarefa = processar_categoria(df_cat, cat, out_file)
        tarefas.append(tarefa)

    # Aguarda as tarefas
    resultados = [t.result() for t in tarefas]
    return resultados

# -------------------------------------------------------------------
# 3) Execução do workflow
# -------------------------------------------------------------------
if __name__ == "__main__":
    resultados = workflow_vendas("vendas.csv")
    print("\n".join(resultados))
