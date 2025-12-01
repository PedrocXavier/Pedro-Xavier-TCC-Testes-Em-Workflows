"""
Workflow Parsl para Análise de Vendas no Varejo
Processa dados de vendas agrupados por categoria em paralelo
"""

import parsl
from parsl.app.app import python_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuração do Parsl com executor de threads para paralelização
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
def carregar_e_agrupar_dados(arquivo_csv):
    """
    Carrega o dataset e retorna lista de categorias únicas
    """
    import pandas as pd
    
    df = pd.read_csv(arquivo_csv)
    
    # Validação das colunas necessárias
    colunas_requeridas = ['produto_id', 'categoria', 'preco', 'quantidade', 'data_venda']
    colunas_faltantes = set(colunas_requeridas) - set(df.columns)
    
    if colunas_faltantes:
        raise ValueError(f"Colunas faltantes no dataset: {colunas_faltantes}")
    
    # Retorna o caminho do arquivo e lista de categorias
    categorias = df['categoria'].unique().tolist()
    
    return {
        'arquivo': arquivo_csv,
        'categorias': categorias,
        'total_registros': len(df)
    }


@python_app
def calcular_estatisticas_categoria(arquivo_csv, categoria):
    """
    Calcula estatísticas para uma categoria específica
    """
    import pandas as pd
    import numpy as np
    
    # Carrega apenas os dados da categoria
    df = pd.read_csv(arquivo_csv)
    df_categoria = df[df['categoria'] == categoria].copy()
    
    # Cálculo das estatísticas
    estatisticas = {
        'categoria': categoria,
        'media_preco': df_categoria['preco'].mean(),
        'desvio_padrao_preco': df_categoria['preco'].std(),
        'total_unidades_vendidas': df_categoria['quantidade'].sum(),
        'receita_total': (df_categoria['preco'] * df_categoria['quantidade']).sum(),
        'numero_vendas': len(df_categoria),
        'preco_minimo': df_categoria['preco'].min(),
        'preco_maximo': df_categoria['preco'].max(),
        'quantidade_media': df_categoria['quantidade'].mean()
    }
    
    return estatisticas


@python_app
def salvar_resultado_categoria(estatisticas, diretorio_saida='resultados'):
    """
    Salva as estatísticas de uma categoria em arquivo CSV
    """
    import pandas as pd
    from pathlib import Path
    
    # Cria diretório de saída se não existir
    Path(diretorio_saida).mkdir(parents=True, exist_ok=True)
    
    categoria = estatisticas['categoria']
    arquivo_saida = f"{diretorio_saida}/resultado_{categoria.replace(' ', '_').lower()}.csv"
    
    # Converte para DataFrame e salva
    df_resultado = pd.DataFrame([estatisticas])
    df_resultado.to_csv(arquivo_saida, index=False, encoding='utf-8')
    
    return {
        'categoria': categoria,
        'arquivo': arquivo_saida,
        'status': 'sucesso'
    }


@python_app
def gerar_relatorio_consolidado(resultados_individuais, diretorio_saida='resultados'):
    """
    Gera um relatório consolidado com todas as categorias
    """
    import pandas as pd
    from pathlib import Path
    
    # Aguarda todos os resultados
    dados_consolidados = []
    
    for resultado in resultados_individuais:
        if resultado['status'] == 'sucesso':
            df_cat = pd.read_csv(resultado['arquivo'])
            dados_consolidados.append(df_cat)
    
    # Combina todos os resultados
    df_consolidado = pd.concat(dados_consolidados, ignore_index=True)
    
    # Ordena por receita total (decrescente)
    df_consolidado = df_consolidado.sort_values('receita_total', ascending=False)
    
    # Salva relatório consolidado
    arquivo_consolidado = f"{diretorio_saida}/relatorio_consolidado.csv"
    df_consolidado.to_csv(arquivo_consolidado, index=False, encoding='utf-8')
    
    return {
        'arquivo_consolidado': arquivo_consolidado,
        'total_categorias': len(df_consolidado),
        'receita_total_geral': df_consolidado['receita_total'].sum()
    }


def executar_workflow(arquivo_entrada='vendas.csv', diretorio_saida='resultados'):
    """
    Função principal que orquestra o workflow
    """
    logger.info("="*60)
    logger.info("Iniciando Workflow de Análise de Vendas")
    logger.info("="*60)
    
    # Etapa 1: Carregar dados e identificar categorias
    logger.info(f"\n[Etapa 1] Carregando dataset: {arquivo_entrada}")
    dados_future = carregar_e_agrupar_dados(arquivo_entrada)
    dados = dados_future.result()
    
    logger.info(f"✓ Dataset carregado: {dados['total_registros']} registros")
    logger.info(f"✓ Categorias encontradas: {len(dados['categorias'])}")
    logger.info(f"  Categorias: {', '.join(dados['categorias'])}")
    
    # Etapa 2: Processar cada categoria em paralelo
    logger.info(f"\n[Etapa 2] Processando categorias em paralelo...")
    
    futures_estatisticas = []
    for categoria in dados['categorias']:
        future = calcular_estatisticas_categoria(arquivo_entrada, categoria)
        futures_estatisticas.append(future)
    
    # Aguarda conclusão dos cálculos
    estatisticas_resultados = [f.result() for f in futures_estatisticas]
    logger.info(f"✓ Estatísticas calculadas para {len(estatisticas_resultados)} categorias")
    
    # Etapa 3: Salvar resultados individuais em paralelo
    logger.info(f"\n[Etapa 3] Salvando resultados individuais...")
    
    futures_salvamento = []
    for estatistica in estatisticas_resultados:
        future = salvar_resultado_categoria(estatistica, diretorio_saida)
        futures_salvamento.append(future)
    
    # Aguarda salvamento
    resultados_salvamento = [f.result() for f in futures_salvamento]
    
    for resultado in resultados_salvamento:
        logger.info(f"✓ Arquivo salvo: {resultado['arquivo']}")
    
    # Etapa 4: Gerar relatório consolidado
    logger.info(f"\n[Etapa 4] Gerando relatório consolidado...")
    
    relatorio_future = gerar_relatorio_consolidado(resultados_salvamento, diretorio_saida)
    relatorio = relatorio_future.result()
    
    logger.info(f"✓ Relatório consolidado: {relatorio['arquivo_consolidado']}")
    
    # Resumo final
    logger.info("\n" + "="*60)
    logger.info("RESUMO DA EXECUÇÃO")
    logger.info("="*60)
    logger.info(f"Total de categorias processadas: {relatorio['total_categorias']}")
    logger.info(f"Receita total geral: R$ {relatorio['receita_total_geral']:,.2f}")
    logger.info(f"Arquivos gerados em: {diretorio_saida}/")
    logger.info("="*60)
    
    return {
        'sucesso': True,
        'categorias_processadas': relatorio['total_categorias'],
        'receita_total': relatorio['receita_total_geral'],
        'diretorio_saida': diretorio_saida
    }


if __name__ == "__main__":
    try:
        # Executa o workflow
        resultado = executar_workflow(
            arquivo_entrada='vendas.csv',
            diretorio_saida='resultados'
        )
        
        print("\n✅ Workflow concluído com sucesso!")
        
    except FileNotFoundError:
        logger.error("❌ Erro: Arquivo 'vendas.csv' não encontrado!")
        logger.error("   Por favor, forneça o arquivo de vendas no mesmo diretório.")
        
    except Exception as e:
        logger.error(f"❌ Erro durante execução: {str(e)}")
        raise
        
    finally:
        # Finaliza o Parsl
        parsl.clear()