"""
Workflow PyCOMPSs para Análise de Vendas no Varejo
Processa dados de vendas agrupados por categoria em paralelo
"""

from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN, FILE_OUT
import pandas as pd
import numpy as np
from collections import defaultdict
import os


@task(returns=dict)
def ler_e_agrupar_dados(arquivo_vendas):
    """
    Lê o dataset de vendas e agrupa por categoria.
    Retorna um dicionário com os dados agrupados.
    """
    print(f"Lendo dados de {arquivo_vendas}...")
    df = pd.read_csv(arquivo_vendas)
    
    # Validação básica
    colunas_esperadas = ['produto_id', 'categoria', 'preco', 'quantidade', 'data_venda']
    for col in colunas_esperadas:
        if col not in df.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no dataset")
    
    # Agrupa por categoria
    grupos = {}
    for categoria, grupo_df in df.groupby('categoria'):
        grupos[categoria] = grupo_df.to_dict('records')
    
    print(f"Total de categorias encontradas: {len(grupos)}")
    return grupos


@task(returns=dict)
def calcular_estatisticas_preco(dados_categoria):
    """
    Calcula média e desvio padrão de preço para uma categoria.
    """
    precos = [item['preco'] for item in dados_categoria]
    
    media_preco = np.mean(precos)
    desvio_preco = np.std(precos, ddof=1)  # desvio padrão amostral
    
    return {
        'media_preco': media_preco,
        'desvio_preco': desvio_preco
    }


@task(returns=int)
def calcular_total_unidades(dados_categoria):
    """
    Calcula o total de unidades vendidas para uma categoria.
    """
    total_unidades = sum(item['quantidade'] for item in dados_categoria)
    return total_unidades


@task(returns=float)
def calcular_receita_total(dados_categoria):
    """
    Calcula a receita total para uma categoria.
    """
    receita = sum(item['preco'] * item['quantidade'] for item in dados_categoria)
    return receita


@task(arquivo_saida=FILE_OUT)
def salvar_resultado_categoria(categoria, dados_categoria, stats_preco, 
                                total_unidades, receita_total, arquivo_saida):
    """
    Consolida os resultados de uma categoria e salva em CSV.
    """
    # Cria DataFrame com os resultados
    resultado = {
        'categoria': [categoria],
        'media_preco': [stats_preco['media_preco']],
        'desvio_preco': [stats_preco['desvio_preco']],
        'total_unidades_vendidas': [total_unidades],
        'receita_total': [receita_total],
        'numero_produtos': [len(dados_categoria)]
    }
    
    df_resultado = pd.DataFrame(resultado)
    df_resultado.to_csv(arquivo_saida, index=False)
    
    print(f"Resultados salvos para categoria '{categoria}' em {arquivo_saida}")


def processar_vendas_paralelo(arquivo_vendas, diretorio_saida='resultados'):
    """
    Função principal que orquestra o workflow paralelo.
    """
    print("="*60)
    print("WORKFLOW PYCOMPSS - ANÁLISE DE VENDAS NO VAREJO")
    print("="*60)
    
    # Cria diretório de saída se não existir
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)
    
    # Etapa 1: Ler e agrupar dados
    grupos_categoria = ler_e_agrupar_dados(arquivo_vendas)
    
    # Etapa 2: Processar cada categoria em paralelo
    resultados = {}
    
    for categoria, dados in grupos_categoria.items():
        print(f"\nProcessando categoria: {categoria}")
        
        # Três tarefas paralelas para cada categoria
        stats_preco = calcular_estatisticas_preco(dados)
        total_unidades = calcular_total_unidades(dados)
        receita = calcular_receita_total(dados)
        
        # Salva resultado da categoria
        arquivo_saida = os.path.join(
            diretorio_saida, 
            f"resultado_{categoria.replace(' ', '_').lower()}.csv"
        )
        
        salvar_resultado_categoria(
            categoria, dados, stats_preco, 
            total_unidades, receita, arquivo_saida
        )
        
        resultados[categoria] = {
            'arquivo': arquivo_saida,
            'stats': stats_preco,
            'unidades': total_unidades,
            'receita': receita
        }
    
    print("\n" + "="*60)
    print("PROCESSAMENTO CONCLUÍDO")
    print("="*60)
    print(f"Arquivos salvos em: {diretorio_saida}/")
    
    return resultados


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    Para executar este workflow:
    
    1. Instale o PyCOMPSs:
       pip install pycompss
    
    2. Execute com o runtime do PyCOMPSs:
       runcompss --python_interpreter=python3 workflow_vendas.py
       
    Ou para execução local simples:
       python workflow_vendas.py
    """
    
    # Arquivo de entrada (fornecido pelo usuário)
    arquivo_vendas = "vendas.csv"
    
    # Verifica se o arquivo existe
    if not os.path.exists(arquivo_vendas):
        print(f"ERRO: Arquivo '{arquivo_vendas}' não encontrado!")
        print("\nCriando dataset de exemplo para demonstração...")
        
        # Cria um dataset de exemplo
        dados_exemplo = {
            'produto_id': range(1, 101),
            'categoria': ['Eletrônicos']*25 + ['Roupas']*25 + ['Alimentos']*25 + ['Livros']*25,
            'preco': np.random.uniform(10, 500, 100),
            'quantidade': np.random.randint(1, 20, 100),
            'data_venda': pd.date_range('2024-01-01', periods=100, freq='D').strftime('%Y-%m-%d').tolist()
        }
        
        df_exemplo = pd.DataFrame(dados_exemplo)
        df_exemplo.to_csv(arquivo_vendas, index=False)
        print(f"Dataset de exemplo criado: {arquivo_vendas}")
    
    # Executa o workflow
    resultados = processar_vendas_paralelo(arquivo_vendas, diretorio_saida='resultados')
    
    print("\nRESUMO DOS RESULTADOS:")
    print("-" * 60)
    for categoria, info in resultados.items():
        print(f"\nCategoria: {categoria}")
        print(f"  Arquivo: {info['arquivo']}")