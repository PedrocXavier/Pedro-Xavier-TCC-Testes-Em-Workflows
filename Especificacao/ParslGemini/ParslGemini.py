import os
import pandas as pd
from parsl.app.app import python_app
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor # Mudança aqui: Importando ThreadPoolExecutor
from typing import List, Dict, Any

# --- Configuração do Parsl ---

# Configuração simples usando ThreadPoolExecutor para demonstrar
# paralelismo baseado em threads dentro do mesmo processo.
# Ideal para tarefas I/O-bound (como leitura/escrita de CSV) e mais leve para testes locais.
config = Config(
    executors=[
        ThreadPoolExecutor( # Usando ThreadPoolExecutor
            label="threads",
            max_threads=4,  # Define o número máximo de threads para paralelismo
        )
    ]
)

# A API Key para o Google Search não é relevante para este workflow de dados,
# mas é mantida aqui em um comentário por boas práticas se o código fosse interagir
# com APIs externas.
# const apiKey = ""

# --- Funções de Aplicação (App) ---

@python_app(executors=['threads']) # Mudança aqui: Usando o label 'threads'
def analyze_category(category_data: Dict[str, List[Any]], category_name: str, output_dir: str) -> str:
    """
    Calcula as estatísticas de vendas para uma única categoria e salva em um arquivo CSV.
    Esta função será executada em paralelo pelo Parsl para cada categoria.

    Args:
        category_data: Dados da categoria em formato dicionário (records).
        category_name: Nome da categoria.
        output_dir: Diretório onde o arquivo CSV de saída será salvo.

    Returns:
        O caminho completo para o arquivo CSV de saída.
    """
    try:
        # Reconstroi o DataFrame a partir dos dados passados
        df = pd.DataFrame.from_records(category_data['data'])

        # 1. Calcular média e desvio padrão de 'preco'
        preco_stats = df['preco'].agg(['mean', 'std']).rename(category_name)
        
        # 2. Calcular total de unidades vendidas
        total_quantidade = df['quantidade'].sum()
        
        # 3. Calcular receita total (já calculada na função principal)
        # Assumindo que 'receita_total' foi incluída nos dados
        if 'receita_total' not in df.columns:
             df['receita_total'] = df['preco'] * df['quantidade']
             
        total_receita = df['receita_total'].sum()

        # Montar o resultado
        result_df = pd.DataFrame({
            'Categoria': [category_name],
            'Preco_Medio': [preco_stats['mean']],
            'Preco_Desvio_Padrao': [preco_stats['std']],
            'Unidades_Vendidas_Total': [total_quantidade],
            'Receita_Total': [total_receita]
        })

        # Cria o diretório de saída se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        output_filepath = os.path.join(output_dir, f"{category_name.replace(' ', '_')}_vendas.csv")
        result_df.to_csv(output_filepath, index=False, sep=';', decimal=',')
        
        return f"Processamento da categoria '{category_name}' concluído. Arquivo salvo em: {output_filepath}"
    
    except Exception as e:
        return f"Erro ao processar categoria '{category_name}': {e}"


# --- Funções Auxiliares e Workflow Principal ---

def generate_mock_data(filepath: str, num_records: int = 1000) -> None:
    """Gera um dataset de vendas simulado."""
    data = {
        'produto_id': range(1, num_records + 1),
        'categoria': [f"Categoria_{i % 5 + 1}" for i in range(num_records)],
        'preco': [round(abs(10 + i % 100 * 0.5 + i % 10 * 0.1), 2) for i in range(num_records)],
        'quantidade': [i % 5 + 1 for i in range(num_records)],
        'data_venda': pd.to_datetime('2023-01-01') + pd.to_timedelta([i * 3 for i in range(num_records)], unit='h')
    }
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Dataset de simulação gerado em: {filepath}")

def run_workflow(input_filepath: str, output_directory: str = "resultados_por_categoria") -> None:
    """
    Função principal que coordena o workflow de processamento de vendas.
    """
    import parsl
    
    try:
        # Inicializa o Parsl
        parsl.load(config)
        print("Parsl inicializado.")

        # 1. Leitura do Dataset
        print(f"Lendo o dataset de vendas: {input_filepath}")
        df_vendas = pd.read_csv(input_filepath)

        # 2. Pré-processamento: Calcular 'receita_total' (preco * quantidade)
        df_vendas['receita_total'] = df_vendas['preco'] * df_vendas['quantidade']
        
        # 3. Agrupamento dos dados por 'categoria'
        grupos = df_vendas.groupby('categoria')
        print(f"Dataset agrupado em {len(grupos)} categorias.")

        futures = []

        # 4. Lançamento das Aplicações Parsl em Paralelo
        for categoria, dados_categoria in grupos:
            print(f"Lançando análise paralela para a categoria: {categoria}...")
            
            # Converte o sub-DataFrame para um formato serializável (dicionário de records)
            # para ser enviado eficientemente ao worker
            dados_serializaveis = {'data': dados_categoria.to_dict('records')}
            
            # Envia a tarefa para execução paralela
            future = analyze_category(dados_serializaveis, categoria, output_directory)
            futures.append(future)

        # 5. Esperar pela conclusão de todas as tarefas e coletar resultados
        print("\nEsperando que todas as tarefas paralelas sejam concluídas...")
        results = [f.result() for f in futures]

        # Exibe os resultados
        print("\n--- Relatório de Conclusão das Tarefas ---")
        for result in results:
            print(result)

        print(f"\nWorkflow concluído. Arquivos de saída salvos no diretório: {output_directory}")

    except Exception as e:
        print(f"Um erro ocorreu durante o workflow: {e}")
    finally:
        # Desliga o Parsl (boa prática)
        parsl.clear()
        
# --- Execução Principal ---
if __name__ == "__main__":
    DATASET_FILENAME = "vendas.csv"
    
    # 1. Gera dados de simulação (se o arquivo não existir)
    if not os.path.exists(DATASET_FILENAME):
        generate_mock_data(DATASET_FILENAME, num_records=5000)

    # 2. Executa o workflow
    run_workflow(DATASET_FILENAME)