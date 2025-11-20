"""
Workflow Parsl para análise paralela de dados de vendas no varejo
Processa dados agrupados por categoria e calcula métricas estatísticas
"""

import parsl
from parsl.app.app import python_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuração do Parsl com executor de threads para I/O
config = Config(
    executors=[
        ThreadPoolExecutor(
            max_threads=8,
            label='threads'
        )
    ]
)

parsl.load(config)


@python_app
def ler_dataset(caminho_arquivo: str) -> pd.DataFrame:
    """
    Lê o dataset de vendas do arquivo CSV
    
    Args:
        caminho_arquivo: Caminho para o arquivo CSV
        
    Returns:
        DataFrame com os dados de vendas
    """
    import pandas as pd
    
    df = pd.read_csv(caminho_arquivo)
    
    # Validação das colunas necessárias
    colunas_requeridas = ['produto_id', 'categoria', 'preco', 'quantidade', 'data_venda']
    if not all(col in df.columns for col in colunas_requeridas):
        raise ValueError(f"Dataset deve conter as colunas: {colunas_requeridas}")
    
    # Conversão de tipos
    df['preco'] = pd.to_numeric(df['preco'], errors='coerce')
    df['quantidade'] = pd.to_numeric(df['quantidade'], errors='coerce')
    df['data_venda'] = pd.to_datetime(df['data_venda'], errors='coerce')
    
    # Remove linhas com valores nulos
    df = df.dropna()
    
    return df


@python_app
def agrupar_por_categoria(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Agrupa o dataset por categoria
    
    Args:
        df: DataFrame com dados de vendas
        
    Returns:
        Dicionário com DataFrames por categoria
    """
    import pandas as pd
    
    grupos = {}
    for categoria, grupo_df in df.groupby('categoria'):
        grupos[categoria] = grupo_df.copy()
    
    return grupos


@python_app
def calcular_estatisticas_preco(df_categoria: pd.DataFrame, categoria: str) -> Dict:
    """
    Calcula média e desvio padrão do preço para uma categoria
    
    Args:
        df_categoria: DataFrame filtrado por categoria
        categoria: Nome da categoria
        
    Returns:
        Dicionário com estatísticas de preço
    """
    import numpy as np
    
    preco_medio = df_categoria['preco'].mean()
    preco_desvio = df_categoria['preco'].std()
    
    return {
        'categoria': categoria,
        'preco_medio': preco_medio,
        'preco_desvio_padrao': preco_desvio
    }


@python_app
def calcular_unidades_vendidas(df_categoria: pd.DataFrame, categoria: str) -> Dict:
    """
    Calcula o total de unidades vendidas para uma categoria
    
    Args:
        df_categoria: DataFrame filtrado por categoria
        categoria: Nome da categoria
        
    Returns:
        Dicionário com total de unidades
    """
    total_unidades = df_categoria['quantidade'].sum()
    
    return {
        'categoria': categoria,
        'total_unidades_vendidas': total_unidades
    }


@python_app
def calcular_receita_total(df_categoria: pd.DataFrame, categoria: str) -> Dict:
    """
    Calcula a receita total para uma categoria
    
    Args:
        df_categoria: DataFrame filtrado por categoria
        categoria: Nome da categoria
        
    Returns:
        Dicionário com receita total
    """
    df_categoria_copy = df_categoria.copy()
    df_categoria_copy['receita'] = df_categoria_copy['preco'] * df_categoria_copy['quantidade']
    receita_total = df_categoria_copy['receita'].sum()
    
    return {
        'categoria': categoria,
        'receita_total': receita_total
    }


@python_app
def consolidar_metricas(
    stats_preco: Dict,
    stats_unidades: Dict,
    stats_receita: Dict
) -> Dict:
    """
    Consolida todas as métricas calculadas em um único dicionário
    
    Args:
        stats_preco: Estatísticas de preço
        stats_unidades: Estatísticas de unidades
        stats_receita: Estatísticas de receita
        
    Returns:
        Dicionário consolidado com todas as métricas
    """
    resultado = {**stats_preco, **stats_unidades, **stats_receita}
    return resultado


@python_app
def salvar_resultado_csv(
    metricas: Dict,
    df_categoria: pd.DataFrame,
    diretorio_saida: str
) -> str:
    """
    Salva os resultados em arquivo CSV
    
    Args:
        metricas: Dicionário com métricas calculadas
        df_categoria: DataFrame da categoria
        diretorio_saida: Diretório onde salvar os arquivos
        
    Returns:
        Caminho do arquivo salvo
    """
    import pandas as pd
    from pathlib import Path
    
    Path(diretorio_saida).mkdir(parents=True, exist_ok=True)
    
    categoria = metricas['categoria']
    nome_arquivo = f"{diretorio_saida}/categoria_{categoria.replace(' ', '_')}.csv"
    
    # Cria DataFrame de resumo
    resumo_df = pd.DataFrame([metricas])
    
    # Adiciona dados detalhados da categoria
    df_detalhado = df_categoria.copy()
    df_detalhado['receita'] = df_detalhado['preco'] * df_detalhado['quantidade']
    
    # Salva arquivo com duas seções: resumo e detalhes
    with open(nome_arquivo, 'w') as f:
        f.write("# RESUMO DA CATEGORIA\n")
        resumo_df.to_csv(f, index=False)
        f.write("\n# DADOS DETALHADOS\n")
        df_detalhado.to_csv(f, index=False)
    
    return nome_arquivo


def processar_workflow_vendas(caminho_dataset: str, diretorio_saida: str = 'resultados'):
    """
    Função principal que orquestra o workflow completo
    
    Args:
        caminho_dataset: Caminho para o arquivo CSV com dados de vendas
        diretorio_saida: Diretório onde salvar os resultados
        
    Returns:
        Lista de futures com caminhos dos arquivos salvos
    """
    logger.info("Iniciando workflow de análise de vendas")
    
    # 1. Ler dataset
    df_future = ler_dataset(caminho_dataset)
    
    # 2. Agrupar por categoria
    grupos_future = agrupar_por_categoria(df_future)
    
    # Aguarda o agrupamento para obter as categorias
    grupos = grupos_future.result()
    
    logger.info(f"Processando {len(grupos)} categorias em paralelo")
    
    # 3. Processar cada categoria em paralelo
    arquivos_salvos = []
    
    for categoria, df_categoria in grupos.items():
        logger.info(f"Processando categoria: {categoria}")
        
        # Calcula métricas em paralelo
        stats_preco_future = calcular_estatisticas_preco(df_categoria, categoria)
        stats_unidades_future = calcular_unidades_vendidas(df_categoria, categoria)
        stats_receita_future = calcular_receita_total(df_categoria, categoria)
        
        # Consolida métricas
        metricas_future = consolidar_metricas(
            stats_preco_future,
            stats_unidades_future,
            stats_receita_future
        )
        
        # Salva resultado
        arquivo_future = salvar_resultado_csv(
            metricas_future,
            df_categoria,
            diretorio_saida
        )
        
        arquivos_salvos.append(arquivo_future)
    
    return arquivos_salvos


def criar_dataset_exemplo(caminho: str = 'vendas_exemplo.csv'):
    """
    Cria um dataset de exemplo para teste
    
    Args:
        caminho: Caminho onde salvar o arquivo
    """
    np.random.seed(42)
    
    categorias = ['Eletrônicos', 'Roupas', 'Alimentos', 'Livros', 'Brinquedos']
    
    dados = []
    for i in range(1000):
        dados.append({
            'produto_id': f'PROD_{i:04d}',
            'categoria': np.random.choice(categorias),
            'preco': np.random.uniform(10, 500),
            'quantidade': np.random.randint(1, 20),
            'data_venda': pd.date_range('2024-01-01', '2024-12-31', periods=1000)[i]
        })
    
    df = pd.DataFrame(dados)
    df.to_csv(caminho, index=False)
    logger.info(f"Dataset de exemplo criado: {caminho}")
    
    return caminho


# Exemplo de uso
if __name__ == '__main__':
    # Cria dataset de exemplo
    arquivo_vendas = criar_dataset_exemplo()
    
    # Executa o workflow
    futures_resultados = processar_workflow_vendas(
        caminho_dataset=arquivo_vendas,
        diretorio_saida='resultados_vendas'
    )
    
    # Aguarda conclusão e exibe resultados
    print("\n" + "="*60)
    print("WORKFLOW CONCLUÍDO")
    print("="*60)
    
    for future in futures_resultados:
        arquivo_salvo = future.result()
        print(f"✓ Arquivo salvo: {arquivo_salvo}")
    
    print("\n" + "="*60)
    print(f"Total de arquivos gerados: {len(futures_resultados)}")
    print("="*60)
    
    # Limpa recursos do Parsl
    parsl.clear()