import parsl
from parsl import python_app, bash_app, DataFlowKernel, Config
from parsl.executors.threads import ThreadPoolExecutor

import pandas as pd
import os

# ========================
# Configuração do Parsl
# ========================
config = Config(
    executors=[
        ThreadPoolExecutor(
            max_threads=4,  # Ajuste de acordo com sua máquina
            label="local_threads"
        )
    ],
    strategy=None,
)
parsl.load(config)


# ========================
# Funções em paralelo
# ========================

@python_app
def processar_categoria(categoria, df_categoria, output_dir="resultados"):
    import pandas as pd
    import os
    
    # Garantir que o diretório de saída exista
    os.makedirs(output_dir, exist_ok=True)

    # Cálculos
    media_preco = df_categoria['preco'].mean()
    desvio_preco = df_categoria['preco'].std()
    total_unidades = df_categoria['quantidade'].sum()
    receita_total = (df_categoria['preco'] * df_categoria['quantidade']).sum()

    # Montar resultado em DataFrame
    resultado = pd.DataFrame([{
        "categoria": categoria,
        "media_preco": media_preco,
        "desvio_preco": desvio_preco,
        "total_unidades": total_unidades,
        "receita_total": receita_total
    }])

    # Salvar CSV específico da categoria
    output_file = os.path.join(output_dir, f"{categoria}.csv")
    resultado.to_csv(output_file, index=False)

    return output_file


# ========================
# Função principal
# ========================
def main():
    # Exemplo: carregando dataset fictício
    df = pd.read_csv("vendas.csv")

    # Agrupar por categoria
    grupos = df.groupby("categoria")

    # Criar lista de tarefas Parsl
    tarefas = []
    for categoria, df_categoria in grupos:
        tarefas.append(processar_categoria(categoria, df_categoria))

    # Esperar resultados
    arquivos_gerados = [t.result() for t in tarefas]

    print("Arquivos gerados:")
    for arq in arquivos_gerados:
        print(arq)


if __name__ == "__main__":
    main()
