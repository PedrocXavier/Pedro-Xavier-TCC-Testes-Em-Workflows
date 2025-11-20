# Este script demonstra o fluxo de trabalho de I/O (leitura), processamento em paralelo
# (agrupamento por categoria) e I/O (escrita) para o sistema ADIOS2 (conceitualmente).
# Usamos Pandas para a lógica de agregação de dados, pois o ADIOS2 foca em I/O de alto desempenho.

import pandas as pd
import numpy as np
import os
import time

# --- 1. Configuração do Workflow e Parâmetros ---

# Simulação do APP ID no ambiente ADIOS2/HPC
APP_ID = "vendas_analysis_1.0"
INPUT_FILE = "vendas.csv"
OUTPUT_DIR = "categorias_processadas"

# Criar o diretório de saída, se não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Configuração do Workflow: APP_ID='{APP_ID}', Diretório de Saída='{OUTPUT_DIR}'")

# --- 2. Leitura do Dataset (Conceito ADIOS2 I/O) ---
# Em um ambiente ADIOS2, esta etapa usaria 'adios2.open()' para ler dados
# em formato BP de forma eficiente e paralela, distribuindo blocos de dados entre os workers (MPI ranks).
# Aqui, simulamos a leitura de um CSV para Pandas.

try:
    print(f"\n[LEITURA I/O] Lendo o arquivo de entrada: {INPUT_FILE}...")
    # Leitura síncrona, para fins de demonstração
    df = pd.read_csv(INPUT_FILE)
    print(f"Leitura concluída. Total de registros: {len(df)}")
    print("Colunas lidas:", df.columns.tolist())

except FileNotFoundError:
    print(f"ERRO: Arquivo {INPUT_FILE} não encontrado. Certifique-se de que ele foi gerado.")
    exit()
except Exception as e:
    print(f"ERRO durante a leitura: {e}")
    exit()

# --- 3. Preparação dos Dados e Agregação ---

# Cálculo da Receita Total para cada linha
# Nota: Esta é a primeira etapa de processamento antes do agrupamento.
df['receita_total_item'] = df['preco'] * df['quantidade']
print("\n[PROCESSAMENTO] Receita total por item calculada.")

# Função para realizar os cálculos de agregação
def calcular_metricas(group_df):
    """
    Calcula as métricas solicitadas para um grupo de categoria.
    Esta função simula o trabalho realizado em paralelo por um worker/rank ADIOS2/HPC.
    """
    categoria = group_df['categoria'].iloc[0]
    
    # Cálculos solicitados
    media_preco = group_df['preco'].mean()
    desvio_padrao_preco = group_df['preco'].std()
    
    # Se houver apenas 1 item, o desvio padrão é NaN, forçamos 0.0
    if pd.isna(desvio_padrao_preco):
        desvio_padrao_preco = 0.0
        
    total_unidades = group_df['quantidade'].sum()
    receita_total_categoria = group_df['receita_total_item'].sum()
    
    # Estrutura de saída (Dataframe de uma linha)
    resultado = pd.DataFrame([{
        'categoria': categoria,
        'media_preco': f"{media_preco:.2f}",
        'desvio_padrao_preco': f"{desvio_padrao_preco:.2f}",
        'total_unidades_vendidas': int(total_unidades),
        'receita_total_categoria': f"{receita_total_categoria:.2f}"
    }])
    
    return resultado

# Agrupamento e Aplicação da Função
# O 'groupby' cria os grupos, e a aplicação em seguida simula a execução paralela
# do processamento em cada grupo (categoria)
print("[AGRUPAMENTO E CÁLCULO] Agrupando dados por 'categoria' e aplicando cálculos...")

# Nota: O 'groupby().apply()' não garante paralelismo de thread/processo como o Dask/Spark,
# mas executa a lógica de cálculo separadamente para cada grupo, atendendo ao requisito lógico.
resultados_por_categoria = df.groupby('categoria').apply(calcular_metricas).reset_index(drop=True)

print("Cálculos de agregação concluídos para todas as categorias.")
print("Resultado consolidado:")
print(resultados_por_categoria)


# --- 4. Escrita dos Resultados (Conceito ADIOS2 I/O) ---
# Em um ambiente ADIOS2, o resultado seria escrito de volta usando 'adios2.open(mode=adios2.Mode.Write)'
# onde cada rank/worker escreveria sua parte em um único arquivo BP de saída.
# O requisito aqui é salvar CADA GRUPO em um CSV SEPARADO.

print(f"\n[ESCRITA I/O] Salvando cada resultado de categoria como um arquivo CSV separado no diretório '{OUTPUT_DIR}'...")
start_time = time.time()

# Iterar sobre os grupos de categoria no DataFrame de resultados
for index, row in resultados_por_categoria.iterrows():
    categoria = row['categoria']
    
    # Filtrar o DataFrame de resultados para obter apenas a linha da categoria atual
    output_df = pd.DataFrame([row])
    
    # Definir o nome do arquivo de saída
    output_filepath = os.path.join(OUTPUT_DIR, f"{categoria.lower().replace(' ', '_')}_metricas.csv")
    
    # Salvar o DataFrame como CSV (sem o índice do Pandas)
    output_df.to_csv(output_filepath, index=False)
    
    print(f"   -> Salvo: {output_filepath}")

end_time = time.time()

# --- 5. Finalização ---

print("\n[FLUXO CONCLUÍDO]")
print(f"Tempo total de escrita: {end_time - start_time:.4f} segundos.")
print("Verifique o diretório 'categorias_processadas' para os arquivos de saída.")

# Exemplo de conteúdo de um dos arquivos CSV de saída (para fins de demonstração)
if os.path.exists(os.path.join(OUTPUT_DIR, "eletronicos_metricas.csv")):
    print("\nConteúdo do arquivo 'eletronicos_metricas.csv' (Exemplo):")
    with open(os.path.join(OUTPUT_DIR, "eletronicos_metricas.csv"), 'r') as f:
        print(f.read())