from mpi4py import MPI
import adios2
import pandas as pd
import numpy as np
import os

# ============================================================
# Inicialização MPI
# ============================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ============================================================
# Leitura do dataset pelo processo 0
# ============================================================
csv_path = "vendas.csv"   # dataset de entrada

if rank == 0:
    df = pd.read_csv(csv_path)
    categorias = sorted(df["categoria"].unique())
else:
    df = None
    categorias = None

# Broadcast para todos os processos
df = comm.bcast(df, root=0)
categorias = comm.bcast(categorias, root=0)

# Cada processo recebe um subconjunto de categorias
categorias_local = categorias[rank::size]

# ============================================================
# Processamento por categoria
# ============================================================
resultados_locais = []

for cat in categorias_local:
    df_cat = df[df["categoria"] == cat]

    preco_medio = df_cat["preco"].mean()
    preco_std = df_cat["preco"].std()
    total_qtd = df_cat["quantidade"].sum()
    receita_total = (df_cat["preco"] * df_cat["quantidade"]).sum()

    resultados_locais.append({
        "categoria": cat,
        "preco_medio": preco_medio,
        "preco_desvio_padrao": preco_std,
        "total_unidades": total_qtd,
        "receita_total": receita_total
    })

# ============================================================
# Escrita de saída com ADIOS2 (um CSV por categoria)
# ============================================================
output_dir = "resultados_por_categoria"
if rank == 0 and not os.path.exists(output_dir):
    os.makedirs(output_dir)
comm.Barrier()

adios = adios2.ADIOS()
io = adios.declare_io("categoria_writer")

for res in resultados_locais:

    cat = res["categoria"]
    filename = f"{output_dir}/{cat}.csv.bp"  # Saída no formato ADIOS2 BP

    # Variáveis ADIOS2
    io.define_variable("categoria", res["categoria"])
    io.define_variable("preco_medio", res["preco_medio"])
    io.define_variable("preco_desvio_padrao", res["preco_desvio_padrao"])
    io.define_variable("total_unidades", res["total_unidades"])
    io.define_variable("receita_total", res["receita_total"])

    # Escrita
    with io.open(filename, "w") as writer:
        writer.write("categoria", res["categoria"])
        writer.write("preco_medio", res["preco_medio"])
        writer.write("preco_desvio_padrao", res["preco_desvio_padrao"])
        writer.write("total_unidades", res["total_unidades"])
        writer.write("receita_total", res["receita_total"])

# ============================================================
# Conversão opcional de BP → CSV (cada processo converte localmente)
# ============================================================
for res in resultados_locais:
    cat = res["categoria"]
    bp_file = f"{output_dir}/{cat}.csv.bp"
    csv_file = f"{output_dir}/{cat}.csv"

    # Escreve CSV legível pelo usuário
    df_out = pd.DataFrame([res])
    df_out.to_csv(csv_file, index=False)

comm.Barrier()

if rank == 0:
    print("Workflow finalizado com sucesso!")
