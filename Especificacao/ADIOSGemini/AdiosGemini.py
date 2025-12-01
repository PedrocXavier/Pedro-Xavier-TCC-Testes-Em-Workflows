import pandas as pd
import numpy as np
import multiprocessing as mp
import os
import time

# Constante para o nome do arquivo de entrada
INPUT_FILE = "vendas.csv"
OUTPUT_DIR = "saida_vendas_por_categoria"

def calcular_estatisticas(grupo):
    """
    Função de agregação personalizada a ser aplicada no groupby.
    Calcula média e desvio padrão de preço, total de unidades vendidas e 
    receita total.
    """
    # Calcula a receita total para o grupo
    grupo['receita'] = grupo['preco'] * grupo['quantidade']
    
    # Agregação das métricas
    return pd.Series({
        'media_preco': grupo['preco'].mean(),
        'desvio_padrao_preco': grupo['preco'].std() if len(grupo) > 1 else 0.0,
        'unidades_vendidas_total': grupo['quantidade'].sum(),
        'receita_total': grupo['receita'].sum()
    })

def salvar_grupo_em_csv(categoria_nome, dados_categoria):
    """
    Salva os dados agregados de uma única categoria em um arquivo CSV.
    Esta função será executada em paralelo.
    """
    filepath = os.path.join(OUTPUT_DIR, f"{categoria_nome.replace(' ', '_')}_resumo.csv")
    
    # Simula o trabalho pesado de I/O que seria paralelizado pelo ADIOS2
    time.sleep(0.1) 
    
    # A coluna 'categoria' é o índice após o reset_index() na agregação.
    # Usamos header=True e index=False para um CSV limpo.
    dados_categoria.to_csv(filepath, index=False, float_format='%.2f')
    print(f"[{mp.current_process().name}] Salvo: {filepath}")

def executar_workflow():
    """
    Função principal que gerencia o fluxo de trabalho de ponta a ponta.
    """
    # 1. Carregar o Dataset (Leitura de Dados)
    print(f"1. Lendo o arquivo: {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{INPUT_FILE}' não encontrado. Certifique-se de que foi gerado.")
        return
    
    # 2. Pré-cálculo e Agregação
    print("2. Calculando métricas e agrupando por categoria...")
    
    # Garantir que a coluna de data esteja no formato correto (opcional, mas boa prática)
    df['data_venda'] = pd.to_datetime(df['data_venda'])
    
    # Adicionar a coluna de Receita por item (preco * quantidade)
    df['receita_item'] = df['preco'] * df['quantidade']
    
    # Agrupar por 'categoria' e aplicar a função de agregação
    # Usamos o agg() do pandas para calcular todas as métricas em uma única passagem:
    resumo_categorias = df.groupby('categoria').agg(
        # Preço: Média e Desvio Padrão
        media_preco=('preco', 'mean'),
        desvio_padrao_preco=('preco', 'std'),
        # Quantidade: Total de unidades vendidas
        unidades_vendidas_total=('quantidade', 'sum'),
        # Receita: Receita Total
        receita_total=('receita_item', 'sum')
    ).reset_index()
    
    # Tratar NaN em desvio_padrao_preco (ocorre para grupos com 1 item)
    resumo_categorias['desvio_padrao_preco'] = resumo_categorias['desvio_padrao_preco'].fillna(0.0)

    print("\nResumo Agregado:")
    print(resumo_categorias.to_string(index=False, float_format='%.2f'))
    
    # 3. Preparar Diretório de Saída
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"\n3. Criado diretório de saída: {OUTPUT_DIR}")
    
    # 4. Salvar os resultados em paralelo (Simulação do conceito de I/O paralelo do ADIOS2)
    print("\n4. Iniciando salvamento paralelo dos arquivos CSV por categoria...")
    
    # Cria uma lista de tarefas (tuplas de (nome_categoria, DataFrame_categoria))
    # Para o salvamento, precisamos dividir o DataFrame agregado em DataFrames por categoria.
    # No entanto, como já temos o resumo agregado, vamos salvar cada LINHA do resumo.
    
    # Cria os argumentos para a função salvar_grupo_em_csv
    tasks = []
    # Iteramos sobre as linhas do resumo agregado
    for _, row in resumo_categorias.iterrows():
        # Converte a linha de Series para DataFrame para o to_csv
        df_to_save = pd.DataFrame([row]) 
        tasks.append((row['categoria'], df_to_save))

    # Cria o pool de processos
    # O número de processos é o número de categorias a serem salvas
    with mp.Pool(len(tasks)) as pool:
        # Usa pool.starmap para executar a função com múltiplos argumentos em paralelo
        # starmap: Mapeia uma função que aceita múltiplos argumentos.
        pool.starmap(salvar_grupo_em_csv, tasks)

    print("\n5. Workflow concluído. Arquivos CSV de resumo salvos no diretório", OUTPUT_DIR)

if __name__ == "__main__":
    # É necessário que o Pandas e o Numpy estejam instalados (pip install pandas numpy)
    executar_workflow()