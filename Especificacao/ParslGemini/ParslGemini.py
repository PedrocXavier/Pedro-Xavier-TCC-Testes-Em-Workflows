# vendas_workflow.py

import pandas as pd
import numpy as np
import os

# Linha original: from parsl.decorators import python_app. Necessária a correção por módulo desconhecido
from parsl import python_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
from parsl import load, join_app, python_app, clear
from parsl.data_provider.files import File
from datetime import datetime, timedelta

# --- 1. Configuração do Parsl ---
# Usamos o ThreadPoolExecutor para paralelismo em threads
config = Config(
    executors=[
        ThreadPoolExecutor(
            label="local_threads",
            max_threads=4  # Ajuste o número de threads conforme os cores da sua máquina
        )
    ]
)

# Limpa configurações anteriores e carrega a nova
clear()
load(config)

# --- 2. Funções do Workflow (Apps Parsl) ---

@python_app
def criar_dataset(num_linhas=1000, nome_arquivo='dados_vendas.csv'):
    """Gera um DataFrame de vendas de varejo e salva em CSV."""
    
    # Simulação de dados
    np.random.seed(42)
    categorias = ['Eletrônicos', 'Vestuário', 'Alimentos', 'Livros', 'Casa']
    
    data = {
        'produto_id': np.arange(1000, 1000 + num_linhas),
        'categoria': np.random.choice(categorias, num_linhas),
        'preco': np.random.uniform(10.0, 500.0, num_linhas).round(2),
        'quantidade': np.random.randint(1, 10, num_linhas),
        'data_venda': [datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(num_linhas)]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(nome_arquivo, index=False)
    
    print(f"Dataset criado em: {nome_arquivo}")
    return nome_arquivo # Retorna o nome do arquivo para o próximo passo


# Esta função não é um app Parsl, mas sim uma função utilitária sequencial
# para a orquestração do fluxo.
def agrupar_por_categoria(caminho_arquivo):
    """
    Lê o dataset, agrupa por categoria e retorna uma lista de grupos.
    
    No Parsl, é mais eficiente carregar o DF uma vez e passar os dados agrupados
    para os apps paralelos (se o volume de dados permitir a serialização
    ou se usarmos um mecanismo de arquivos intermediários).
    Aqui, faremos a leitura e o agrupamento para dividir o trabalho.
    """
    df = pd.read_csv(caminho_arquivo)
    
    # Garantindo que 'preco' e 'quantidade' sejam numéricos
    df['preco'] = pd.to_numeric(df['preco'])
    df['quantidade'] = pd.to_numeric(df['quantidade'])
    
    grupos = []
    # Cria uma cópia do DataFrame do grupo para evitar side-effects
    for categoria, grupo_df in df.groupby('categoria'):
        grupos.append((categoria, grupo_df.copy()))
        
    print(f"Dados agrupados em {len(grupos)} categorias.")
    return grupos


@python_app
def calcular_estatisticas(categoria, grupo_df, output_dir='output_vendas'):
    """
    Calcula métricas para um grupo de categoria e salva o resultado em CSV.
    Esta função será executada em paralelo para cada categoria.
    """
    
    # Cálculo da Receita Total
    grupo_df['receita'] = grupo_df['preco'] * grupo_df['quantidade']
    
    # Cálculo das Métricas
    media_preco = grupo_df['preco'].mean()
    desvio_padrao_preco = grupo_df['preco'].std()
    total_unidades = grupo_df['quantidade'].sum()
    receita_total = grupo_df['receita'].sum()
    
    # Cria o DataFrame de resultados
    resultado = pd.DataFrame([{
        'categoria': categoria,
        'media_preco': media_preco,
        'desvio_padrao_preco': desvio_padrao_preco,
        'total_unidades_vendidas': total_unidades,
        'receita_total': receita_total
    }])
    
    # Cria o diretório de saída se não existir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Define o nome do arquivo de saída
    nome_saida = os.path.join(output_dir, f'estatisticas_{categoria.lower().replace(" ", "_")}.csv')
    
    # Salva o resultado no arquivo CSV
    resultado.to_csv(nome_saida, index=False)
    
    print(f"Estatísticas de '{categoria}' salvas em: {nome_saida}")
    return nome_saida


# --- 3. Fluxo Principal do Workflow ---

if __name__ == '__main__':
    
    # 1. Geração do Dataset (App Parsl)
    print("Iniciando a geração do dataset...")
    future_caminho_dataset = criar_dataset(num_linhas=50000)
    caminho_dataset = future_caminho_dataset.result()
    
    # 2. Leitura e Agrupamento (Sequencial)
    # Bloqueia a execução até que o arquivo esteja pronto.
    grupos_de_dados = agrupar_por_categoria(caminho_dataset)
    
    # 3. Processamento Paralelo por Categoria (Apps Parsl)
    print("\nIniciando processamento paralelo por categoria...")
    futures_resultados = []
    
    for categoria, grupo_df in grupos_de_dados:
        # Lança a função de cálculo em paralelo para cada categoria
        future = calcular_estatisticas(categoria, grupo_df)
        futures_resultados.append(future)
    
    # 4. Espera pelos resultados e coleta dos caminhos de saída
    # O Parsl gerencia a execução em paralelo de todos os apps 'calcular_estatisticas'
    caminhos_de_saida = [f.result() for f in futures_resultados]
    
    print("\n--- Workflow Concluído ---")
    print("Arquivos de saída gerados:")
    for caminho in caminhos_de_saida:
        print(f"- {caminho}")
    
    # Exemplo de como ler e exibir um dos resultados
    print("\nConteúdo de um arquivo de exemplo ('Livros'):")
    caminho_livros = [c for c in caminhos_de_saida if 'livros' in c][0]
    df_resultado_exemplo = pd.read_csv(caminho_livros)
    print(df_resultado_exemplo)