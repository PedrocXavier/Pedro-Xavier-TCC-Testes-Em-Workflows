import adios2
import pandas as pd
import numpy as np
import os

# ================================================================
# 1. Leitura do dataset
# ================================================================
dataset = pd.read_csv("vendas.csv")

# Agrupar por categoria
categorias = dataset["categoria"].unique()


# ================================================================
# 2. Função que processa cada categoria e escreve via ADIOS2
# ================================================================
def processar_categoria(cat):
    df_cat = dataset[dataset["categoria"] == cat]

    preco = df_cat["preco"].to_numpy()
    quantidade = df_cat["quantidade"].to_numpy()

    media_preco = float(np.mean(preco))
    desvio_preco = float(np.std(preco))
    total_unidades = int(np.sum(quantidade))
    receita_total = float(np.sum(preco * quantidade))

    # ------------------------------------------------------------
    # Escrever os dados usando ADIOS2 (formato BP4)
    # ------------------------------------------------------------
    bpfile = f"saida_{cat}.bp"

    adios = adios2.Adios() # Linha gerada originalmente foi adios = adios2.ADIOS()
    io = adios.declare_io(f"io_{cat}") # Linha gerada originalmente foi io = adios.DeclareIO
    io.set_engine("BP4") # Linha gerada originalmente foi io.SetEngine

    var_media = io.define_variable("media_preco", media_preco) # Linha gerada originalmente foi var_media = io.DefineVariable
    var_std = io.define_variable("desvio_preco", desvio_preco) # Linha gerada originalmente foi var_desvio = io.DefineVariable
    var_units = io.define_variable("total_unidades", total_unidades) # Linha gerada originalmente foi var_unidades = io.DefineVariable
    var_receita = io.define_variable("receita_total", receita_total) # Linha gerada originalmente foi var_receita = io.DefineVariable

    writer = io.open(bpfile, adios2.Mode.Write) # Linha gerada originalmente foi engine = io.Open

    writer.begin_step() # Linha gerada originalmente foi engine.BeginStep()
    writer.put(var_media, media_preco) # Linha gerada originalmente foi writer.Put
    writer.put(var_std, desvio_preco) # Linha gerada originalmente foi writer.Put
    writer.put(var_units, total_unidades) # Linha gerada originalmente foi writer.Put
    writer.put(var_receita, receita_total) # Linha gerada originalmente foi writer.Put
    writer.end_step() # Linha gerada originalmente foi engine.EndStep()

    writer.close() # Linha gerada originalmente foi engine.Close()

    # ------------------------------------------------------------
    # Após gerar o BP4, criar o CSV correspondente
    # ------------------------------------------------------------
    df_out = pd.DataFrame(
        [{
            "categoria": cat,
            "media_preco": media_preco,
            "desvio_preco": desvio_preco,
            "total_unidades": total_unidades,
            "receita_total": receita_total,
        }]
    )

    df_out.to_csv(f"saida_{cat}.csv", index=False)

    print(f"[OK] Categoria '{cat}' processada.")


# ================================================================
# 3. Execução em paralelo (MPI ou Threads)
# ================================================================
# Se você rodar com mpirun/mpiexec, cada processo cuida de uma categoria.
# Exemplo:
#     mpirun -np 4 python workflow.py
# ================================================================
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Distribuir categorias entre processos
    for i, cat in enumerate(categorias):
        if i % size == rank:
            processar_categoria(cat)

    comm.Barrier()

except ImportError:
    # Execução sequencial caso mpi4py não esteja disponível
    for cat in categorias:
        processar_categoria(cat)
