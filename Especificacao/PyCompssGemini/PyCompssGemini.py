import pandas as pd
import os
import random
import numpy as np

# Importa os decorators do PyCOMPSs (Assumindo que o PyCOMPSs está instalado)
# Se o PyCOMPSs não estiver instalado, os decorators serão ignorados e o código será executado sequencialmente.
try:
    from pycompss.api.task import task
    from pycompss.api.parameter import FILE_IN, FILE_OUT, COLLECTION_FILE_OUT
    PYCOMPSS_AVAILABLE = True
except ImportError:
    # Cria stubs para que o código possa ser testado mesmo sem PyCOMPSs
    print("AVISO: O PyCOMPSs não foi encontrado. As tarefas serão executadas sequencialmente.")
    def task(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    FILE_IN = object()
    FILE_OUT = object()
    COLLECTION_FILE_OUT = object()
    PYCOMPSS_AVAILABLE = False


def generate_dummy_data(filename="vendas.csv", num_rows=1000):
    """Gera um arquivo CSV de vendas de exemplo."""
    print(f"Gerando dados simulados para {filename}...")
    categories = ['Eletronicos', 'Vestuario', 'Alimentos', 'Casa']
    
    data = {
        'produto_id': [f'P{i:04d}' for i in range(num_rows)],
        'categoria': [random.choice(categories) for _ in range(num_rows)],
        'preco': [round(random.uniform(5.0, 500.0), 2) for _ in range(num_rows)],
        'quantidade': [random.randint(1, 15) for _ in range(num_rows)],
        'data_venda': pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 365, num_rows), unit='D')
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Arquivo '{filename}' gerado com sucesso.")


@task(input_path=FILE_IN, returns=list)
def split_data(input_path, output_dir="temp_categories"):
    """
    Tarefa PyCOMPSs para ler o dataset, calcular a receita e dividir
    o DataFrame em arquivos CSV separados por categoria.
    """
    print(f"Lendo dados de: {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Calcular Receita
    df['receita'] = df['preco'] * df['quantidade']
    
    # 2. Configurar diretório de saída
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    category_files = []
    
    # 3. Agrupar e salvar cada grupo em um arquivo separado
    grouped = df.groupby('categoria')
    
    for category, data in grouped:
        safe_category_name = category.replace(' ', '_').lower()
        output_filepath = os.path.join(output_dir, f"{safe_category_name}.csv")
        data.to_csv(output_filepath, index=False)
        category_files.append(output_filepath)
        print(f"Arquivo de categoria salvo: {output_filepath}")
        
    return category_files


@task(input_file=FILE_IN, output_file=FILE_OUT)
def analyze_category(input_file, output_file):
    """
    Tarefa PyCOMPSs para calcular métricas de vendas para uma única categoria.
    Esta tarefa será executada em paralelo.
    """
    print(f"Iniciando análise para arquivo: {input_file}")
    df = pd.read_csv(input_file)
    
    # Extrai o nome da categoria do nome do arquivo
    category_name = os.path.basename(input_file).replace('.csv', '').split('_')[-1].capitalize()

    # 1. Calcular Média e Desvio Padrão de Preço
    mean_price = df['preco'].mean()
    std_price = df['preco'].std()
    
    # 2. Total de Unidades Vendidas
    total_units = df['quantidade'].sum()
    
    # 3. Receita Total (A coluna 'receita' foi adicionada na tarefa split_data)
    total_revenue = df['receita'].sum()
    
    # Cria o DataFrame de resultados (uma única linha)
    results = pd.DataFrame({
        'Categoria': [category_name],
        'Preco_Medio': [mean_price],
        'Preco_Desvio_Padrao': [std_price],
        'Unidades_Totais': [total_units],
        'Receita_Total': [total_revenue]
    })
    
    # Salva o resultado no arquivo de saída CSV
    results.to_csv(output_file, index=False)
    print(f"Análise concluída e salva em: {output_file}")


def main_workflow(input_data_path="vendas.csv", final_results_dir="analise_resultados"):
    """
    Função principal do workflow PyCOMPSs.
    """
    print("--- INICIANDO WORKFLOW PYCOMPSS ---")

    # Garante que o diretório de resultados finais exista
    if not os.path.exists(final_results_dir):
        os.makedirs(final_results_dir)

    # 1. Tarefa de Agrupamento: Divide o dataset por categoria.
    # Esta tarefa retorna uma Future Object (lista) com os caminhos dos arquivos.
    category_files_future = split_data(input_data_path)

    # 2. Tarefas Paralelas de Análise
    print("Lançando tarefas de análise em paralelo...")
    
    # O PyCOMPSs aguardará implicitamente o resultado de split_data (category_files_future)
    # antes de iterar e lançar as tarefas analyze_category.
    result_futures = []
    
    # Itera sobre os caminhos dos arquivos gerados (Future List)
    # Aqui, category_files_future atua como um objeto PyCOMPSs que se resolverá antes da iteração.
    for category_file_path in category_files_future:
        # Define o nome do arquivo de saída final
        filename = os.path.basename(category_file_path)
        category_name = filename.replace('.csv', '')
        output_filepath = os.path.join(final_results_dir, f"analise_{category_name}.csv")
        
        # Lança a tarefa de análise em paralelo
        # O output_filepath é decorado implicitamente como um FILE_OUT no decorate
        future = analyze_category(category_file_path, output_filepath)
        result_futures.append(future)

    # 3. Sincronização e Conclusão
    # Se estivéssemos executando com `runcompss`, o workflow aguardaria automaticamente
    # a conclusão de todos os Future Objects (result_futures).
    # Como não temos uma tarefa final que os consuma, este loop é mais para ilustrar
    # que todos os resultados foram lançados e o PyCOMPSs irá garantir que os
    # arquivos de saída sejam gerados.
    
    print("\nTodas as tarefas de análise foram lançadas.")
    
    if PYCOMPSS_AVAILABLE:
        print("Aguardando a conclusão de todas as tarefas...")
        # A sincronização final pode ser forçada explicitamente se necessário,
        # mas muitas vezes é implícita no final do programa PyCOMPSs.
        # from pycompss.api.api import compss_wait_on
        # _ = compss_wait_on(result_futures)
        print("Resultados salvos individualmente. O workflow PyCOMPSs foi concluído.")
    else:
        # Se não estiver rodando com PyCOMPSs, os resultados já estão prontos.
        print("Resultados salvos individualmente.")

    print(f"Verifique o diretório '{final_results_dir}' para os arquivos de resultados.")
    print("--- FIM DO WORKFLOW ---")


if __name__ == "__main__":
    # Define os caminhos dos arquivos
    INPUT_FILE = "vendas.csv"
    
    # 1. Prepara o ambiente e os dados
    generate_dummy_data(INPUT_FILE)

    # 2. Executa o workflow principal
    # Para executar em um ambiente PyCOMPSs real, você usaria o comando:
    # runcompss retail_analysis_workflow.py
    
    if PYCOMPSS_AVAILABLE:
        # Se for executado via `runcompss`, ele usará o módulo principal
        main_workflow(INPUT_FILE)
    else:
        # Se for executado diretamente, simula o fluxo sequencial (para teste)
        main_workflow(INPUT_FILE)

    # Nota: Em um ambiente PyCOMPSs real, o diretório "temp_categories"
    # seria limpo automaticamente ou gerenciado pelo runtime.