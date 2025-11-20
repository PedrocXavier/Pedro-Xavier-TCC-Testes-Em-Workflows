"""
Workflow PyCOMPSs para análise paralela de dados de vendas no varejo
Agrupa por categoria e calcula estatísticas em paralelo
"""

from pycompss.api.task import task
from pycompss.api.parameter import IN, FILE_OUT
from pycompss.api.api import compss_wait_on, compss_barrier
import pandas as pd
import numpy as np
from pathlib import Path
import os


@task(returns=dict)
def ler_dataset(arquivo_csv):
    """
    Lê o dataset de vendas e retorna um dicionário agrupado por categoria
    """
    df = pd.read_csv(arquivo_csv)
    
    # Converte data_venda para datetime se necessário
    if 'data_venda' in df.columns:
        df['data_venda'] = pd.to_datetime(df['data_venda'])
    
    # Agrupa os dados por categoria
    grupos = {}
    for categoria in df['categoria'].unique():
        grupos[categoria] = df[df['categoria'] == categoria].to_dict('records')
    
    return grupos


@task(returns=dict)
def calcular_estatisticas_preco(dados_categoria, categoria):
    """
    Calcula média e desvio padrão de preço para uma categoria
    """
    df = pd.DataFrame(dados_categoria)
    
    estatisticas = {
        'categoria': categoria,
        'preco_medio': df['preco'].mean(),
        'preco_desvio_padrao': df['preco'].std(),
        'preco_minimo': df['preco'].min(),
        'preco_maximo': df['preco'].max()
    }
    
    return estatisticas


@task(returns=dict)
def calcular_volume_vendas(dados_categoria, categoria):
    """
    Calcula total de unidades vendidas para uma categoria
    """
    df = pd.DataFrame(dados_categoria)
    
    volume = {
        'categoria': categoria,
        'total_unidades_vendidas': df['quantidade'].sum(),
        'numero_transacoes': len(df),
        'quantidade_media_por_venda': df['quantidade'].mean()
    }
    
    return volume


@task(returns=dict)
def calcular_receita(dados_categoria, categoria):
    """
    Calcula receita total para uma categoria
    """
    df = pd.DataFrame(dados_categoria)
    
    # Calcula receita por transação
    df['receita'] = df['preco'] * df['quantidade']
    
    receita = {
        'categoria': categoria,
        'receita_total': df['receita'].sum(),
        'receita_media_transacao': df['receita'].mean(),
        'ticket_medio': (df['preco'] * df['quantidade']).sum() / len(df)
    }
    
    return receita


@task(arquivo_saida=FILE_OUT)
def salvar_resultados_categoria(stats_preco, stats_volume, stats_receita, 
                                 dados_categoria, categoria, arquivo_saida):
    """
    Consolida todos os resultados de uma categoria e salva em CSV
    """
    df = pd.DataFrame(dados_categoria)
    
    # Cria DataFrame com estatísticas consolidadas
    resultados = {
        'Categoria': [categoria],
        'Preço Médio': [stats_preco['preco_medio']],
        'Desvio Padrão Preço': [stats_preco['preco_desvio_padrao']],
        'Preço Mínimo': [stats_preco['preco_minimo']],
        'Preço Máximo': [stats_preco['preco_maximo']],
        'Total Unidades Vendidas': [stats_volume['total_unidades_vendidas']],
        'Número de Transações': [stats_volume['numero_transacoes']],
        'Quantidade Média por Venda': [stats_volume['quantidade_media_por_venda']],
        'Receita Total': [stats_receita['receita_total']],
        'Receita Média por Transação': [stats_receita['receita_media_transacao']],
        'Ticket Médio': [stats_receita['ticket_medio']]
    }
    
    df_resultado = pd.DataFrame(resultados)
    
    # Salva no arquivo
    df_resultado.to_csv(arquivo_saida, index=False, encoding='utf-8')
    
    print(f"✓ Resultados salvos: {arquivo_saida}")


def processar_categoria(dados_categoria, categoria, diretorio_saida):
    """
    Processa uma categoria em paralelo: calcula todas as estatísticas
    """
    # Dispara tarefas paralelas para cálculos diferentes
    stats_preco = calcular_estatisticas_preco(dados_categoria, categoria)
    stats_volume = calcular_volume_vendas(dados_categoria, categoria)
    stats_receita = calcular_receita(dados_categoria, categoria)
    
    # Define arquivo de saída
    nome_arquivo = f"{categoria.replace(' ', '_').replace('/', '_')}_analise.csv"
    arquivo_saida = os.path.join(diretorio_saida, nome_arquivo)
    
    # Salva resultados consolidados
    salvar_resultados_categoria(stats_preco, stats_volume, stats_receita,
                                dados_categoria, categoria, arquivo_saida)
    
    return arquivo_saida


def main():
    """
    Função principal do workflow
    """
    # Configurações
    arquivo_entrada = 'vendas.csv' 
    diretorio_saida = 'resultados_analise'
    
    # Cria diretório de saída se não existir
    Path(diretorio_saida).mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("WORKFLOW PYCOMPSS - ANÁLISE DE VENDAS NO VAREJO")
    print("=" * 60)
    print(f"\n📁 Lendo dataset: {arquivo_entrada}")
    
    # Passo 1: Ler e agrupar dados por categoria
    grupos_categorias = ler_dataset(arquivo_entrada)
    
    # Aguarda a leitura e agrupamento
    grupos_categorias = compss_wait_on(grupos_categorias)
    
    print(f"\n✓ Dataset carregado e agrupado")
    print(f"✓ Categorias encontradas: {len(grupos_categorias)}")
    print(f"  Categorias: {', '.join(grupos_categorias.keys())}\n")
    
    # Passo 2: Processar cada categoria em paralelo
    print("🚀 Iniciando processamento paralelo das categorias...\n")
    
    arquivos_saida = []
    for categoria, dados in grupos_categorias.items():
        print(f"  → Processando categoria: {categoria}")
        arquivo = processar_categoria(dados, categoria, diretorio_saida)
        arquivos_saida.append(arquivo)
    
    # Aguarda todas as tarefas terminarem
    compss_barrier()
    
    print("\n" + "=" * 60)
    print("✅ PROCESSAMENTO CONCLUÍDO!")
    print("=" * 60)
    print(f"\n📊 Arquivos gerados em: {diretorio_saida}/")
    print(f"   Total de arquivos: {len(arquivos_saida)}")
    print("\nCada arquivo contém:")
    print("  • Estatísticas de preço (média, desvio padrão, min, max)")
    print("  • Volume de vendas (total unidades, transações)")
    print("  • Receita (total, média, ticket médio)")
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()