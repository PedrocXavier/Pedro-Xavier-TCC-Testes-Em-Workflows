import pandas as pd
import numpy as np
import os
from pycompss.api.task import task
from pycompss.api.api import compss_wait_on
from pycompss.api.parameter import FILE_OUT

# ==============================================================================
# Funções Auxiliares (Tarefas PyCOMPSs)
# ==============================================================================

@task(returns=pd.DataFrame)
def ler_dataset(filepath):
    """
    Tarefa para ler o dataset de vendas.
    Em um cenário real, esta função leria um arquivo grande.
    """
    print(f"Lendo dataset de: {filepath}")
    # Simula a leitura de um arquivo CSV, mas usa a função de criação
    # para garantir que os dados de simulação estejam disponíveis.
    # No PyCOMPSs, se for um arquivo persistente, você usaria pd.read_csv(filepath)
    # Se você estiver usando um dataset de simulação, pode ser mais fácil criá-lo
    # dentro da tarefa ou passá-lo como um objeto. Vamos simular a leitura.
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"ERRO: Arquivo {filepath} não encontrado. Certifique-se de que a função 'criar_dataset_simulado' foi executada.")
        return pd.DataFrame()

@task(returns=pd.DataFrame)
def agrupar_por_categoria(df):
    """
    Tarefa para agrupar o DataFrame e retornar um dicionário de DataFrames
    onde a chave é a categoria.
    """
    print("Agrupando dados por categoria...")
    grupos = {}
    for categoria, grupo_df in df.groupby('categoria'):
        grupos[categoria] = grupo_df.copy()  # Cria cópia para isolar dados da tarefa
    return grupos

@task(outfile=FILE_OUT)
def calcular_e_salvar_analise(categoria, df_grupo, outfile):
    """
    Tarefa para calcular as métricas e salvar o resultado em um arquivo CSV.
    Esta tarefa é executada em paralelo para cada categoria.
    O decorador FILE_OUT informa ao PyCOMPSs que esta tarefa produz um arquivo.
    """
    print(f"Calculando métricas para a categoria: {categoria}")

    # 1. Média e Desvio Padrão do Preço
    media_preco = df_grupo['preco'].mean()
    desvio_padrao_preco = df_grupo['preco'].std()

    # 2. Total de Unidades Vendidas (Soma da 'quantidade')
    total_unidades = df_grupo['quantidade'].sum()

    # 3. Receita Total
    df_grupo['receita'] = df_grupo['preco'] * df_grupo['quantidade']
    receita_total = df_grupo['receita'].sum()

    # Cria o DataFrame de resultado
    resultado = pd.DataFrame({
        'Categoria': [categoria],
        'Media_Preco': [media_preco],
        'Desvio_Padrao_Preco': [desvio_padrao_preco],
        'Total_Unidades_Vendidas': [total_unidades],
        'Receita_Total': [receita_total]
    })

    # Salva o resultado no arquivo CSV
    resultado.to_csv(outfile, index=False)
    print(f"Análise da categoria '{categoria}' salva em: {outfile}")

# ==============================================================================
# Função Principal (Workflow)
# ==============================================================================

def workflow_analise_vendas(input_file, output_dir="resultados_analise"):
    """
    Define o fluxo principal de execução PyCOMPSs.
    """
    print("Iniciando o Workflow de Análise de Vendas com PyCOMPSs.")

    # 1. Leitura do Dataset (Tarefa 1)
    df_future = ler_dataset(input_file)

    # 2. Agrupamento (Tarefa 2)
    # PyCOMPSs automaticamente gerencia a dependência de dados
    grupos_future = agrupar_por_categoria(df_future)

    # Espera que o agrupamento termine para obter as chaves (categorias)
    # mas os DataFrames de cada grupo continuam como objetos futuros.
    grupos = compss_wait_on(grupos_future)
    
    # Lista de resultados futuros (para sincronização final)
    resultados_futuros = []

    # 3. Processamento Paralelo por Categoria (Tarefa 3 em Paralelo)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for categoria, df_grupo_future in grupos.items():
        # Define o nome do arquivo de saída
        output_filename = os.path.join(output_dir, f"{categoria.replace(' ', '_')}_analise.csv")
        
        # Envia a tarefa de cálculo e salvamento.
        # df_grupo_future é o DataFrame do grupo (ainda um objeto futuro)
        # O PyCOMPSs garante que o cálculo só ocorra após o df_grupo_future estar pronto.
        resultado_future = calcular_e_salvar_analise(categoria, df_grupo_future, output_filename)
        resultados_futuros.append(resultado_future)

    # 4. Sincronização Final
    # Espera que todas as tarefas de salvamento terminem
    compss_wait_on(resultados_futuros)

    print("Workflow de análise concluído. Resultados salvos na pasta:", output_dir)
    return True

# ==============================================================================
# Execução Principal e Simulação de Dados
# ==============================================================================

def criar_dataset_simulado(filepath, num_linhas=10000):
    """ Cria um dataset simulado para testar o workflow. """
    print(f"Criando dataset de simulação com {num_linhas} linhas...")
    np.random.seed(42)
    
    categorias = ['Eletronicos', 'Roupas', 'Alimentos', 'Livros', 'Casa']
    
    data = {
        'produto_id': np.arange(num_linhas),
        'categoria': np.random.choice(categorias, num_linhas),
        'preco': np.random.randint(10, 500, num_linhas) * np.random.rand(num_linhas),
        'quantidade': np.random.randint(1, 20, num_linhas),
        'data_venda': pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 365, num_linhas), unit='D')
    }
    
    df = pd.DataFrame(data)
    # Arredonda o preço para duas casas decimais
    df['preco'] = df['preco'].round(2) 
    df.to_csv(filepath, index=False)
    print(f"Dataset simulado salvo em: {filepath}")
    return filepath

if __name__ == '__main__':
    # Nome do arquivo de dataset simulado
    INPUT_CSV = "vendas_dataset.csv"

    # Cria o dataset antes de iniciar o PyCOMPSs
    criar_dataset_simulado(INPUT_CSV)
    
    # Execução do workflow (normalmente através do runcompss)
    # Para testes em um ambiente sequencial (sem o runcompss),
    # você precisaria usar o API 'compss' para inicializar e finalizar.
    # No entanto, o modo de execução correto para PyCOMPSs é via runcompss.
    
    # Se você quiser testar diretamente via python, descomente as linhas abaixo,
    # mas o paralelismo real será perdido sem o runtime do COMPSs:
    # from pycompss.api.api import compss_start, compss_stop
    # compss_start()
    # workflow_analise_vendas(INPUT_CSV)
    # compss_stop()

    print("\n--- INSTRUÇÕES DE EXECUÇÃO ---")
    print("Para executar este workflow com o PyCOMPSs, use o comando:")
    print(f"runcompss analise_vendas.py")

    print("\nOs resultados serão salvos na pasta 'resultados_analise'.")