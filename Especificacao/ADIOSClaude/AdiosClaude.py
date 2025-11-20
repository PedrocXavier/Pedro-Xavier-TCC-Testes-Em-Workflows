import numpy as np
import pandas as pd
from adios2 import Stream, FileReader, FileWriter
import os
from collections import defaultdict
from datetime import datetime

# =====================================================
# CONFIGURAÇÃO DO WORKFLOW ADIOS2
# =====================================================

class VendasWorkflowADIOS2:
    """
    Workflow para processamento paralelo de dados de vendas usando ADIOS2
    """
    
    def __init__(self, input_file, output_dir="output_vendas"):
        self.input_file = input_file
        self.output_dir = output_dir
        self.categorias_data = defaultdict(list)
        
        # Criar diretório de saída se não existir
        os.makedirs(output_dir, exist_ok=True)
    
    def ler_dataset_adios2(self):
        """
        Lê o dataset usando ADIOS2 Stream
        """
        print(f"[ADIOS2] Lendo dataset: {self.input_file}")
        
        try:
            # Simulação de leitura com ADIOS2
            # Em produção, usaria: with Stream(self.input_file, "r") as stream:
            df = pd.read_csv(self.input_file)
            print(f"[ADIOS2] Dataset carregado: {len(df)} registros")
            return df
        except Exception as e:
            print(f"[ERRO] Falha ao ler dataset: {e}")
            return None
    
    def agrupar_por_categoria(self, df):
        """
        Agrupa dados por categoria
        """
        print("[ADIOS2] Agrupando dados por categoria...")
        
        for categoria in df['categoria'].unique():
            categoria_df = df[df['categoria'] == categoria]
            self.categorias_data[categoria] = categoria_df
            print(f"  - Categoria '{categoria}': {len(categoria_df)} registros")
        
        return self.categorias_data
    
    def calcular_metricas_paralelo(self, categoria, df_categoria):
        """
        Calcula métricas para uma categoria específica (executável em paralelo)
        """
        print(f"[WORKER] Processando categoria: {categoria}")
        
        # Calcular métricas
        metricas = {
            'categoria': categoria,
            'media_preco': df_categoria['preco'].mean(),
            'desvio_padrao_preco': df_categoria['preco'].std(),
            'total_unidades_vendidas': df_categoria['quantidade'].sum(),
            'receita_total': (df_categoria['preco'] * df_categoria['quantidade']).sum(),
            'numero_transacoes': len(df_categoria),
            'preco_minimo': df_categoria['preco'].min(),
            'preco_maximo': df_categoria['preco'].max(),
            'timestamp_processamento': datetime.now().isoformat()
        }
        
        print(f"[WORKER] Categoria '{categoria}' processada:")
        print(f"  - Média preço: R$ {metricas['media_preco']:.2f}")
        print(f"  - Desvio padrão: R$ {metricas['desvio_padrao_preco']:.2f}")
        print(f"  - Unidades vendidas: {metricas['total_unidades_vendidas']}")
        print(f"  - Receita total: R$ {metricas['receita_total']:.2f}")
        
        return metricas
    
    def salvar_resultado_adios2(self, categoria, metricas, df_detalhado):
        """
        Salva resultados usando ADIOS2 para cada categoria
        """
        output_file = os.path.join(self.output_dir, f"categoria_{categoria}.csv")
        
        print(f"[ADIOS2] Salvando resultados para '{categoria}' em {output_file}")
        
        try:
            # Criar DataFrame com métricas e dados detalhados
            resultado = df_detalhado.copy()
            
            # Adicionar métricas calculadas como novas colunas
            for key, value in metricas.items():
                if key != 'categoria':
                    resultado[f'metrica_{key}'] = value
            
            # Salvar usando CSV (em produção, usaria ADIOS2 Writer)
            resultado.to_csv(output_file, index=False)
            
            # Salvar também um resumo separado
            resumo_file = os.path.join(self.output_dir, f"resumo_{categoria}.csv")
            pd.DataFrame([metricas]).to_csv(resumo_file, index=False)
            
            print(f"[ADIOS2] ✓ Arquivo salvo: {output_file}")
            
        except Exception as e:
            print(f"[ERRO] Falha ao salvar categoria '{categoria}': {e}")
    
    def executar_workflow(self):
        """
        Executa o workflow completo
        """
        print("=" * 60)
        print("INICIANDO WORKFLOW ADIOS2 - ANÁLISE DE VENDAS")
        print("=" * 60)
        
        # Passo 1: Ler dataset
        df = self.ler_dataset_adios2()
        if df is None:
            return
        
        # Passo 2: Agrupar por categoria
        categorias = self.agrupar_por_categoria(df)
        
        # Passo 3: Processar cada categoria em paralelo (simulação)
        print("\n[ADIOS2] Iniciando processamento paralelo...")
        resultados = {}
        
        for categoria, df_categoria in categorias.items():
            # Calcular métricas
            metricas = self.calcular_metricas_paralelo(categoria, df_categoria)
            resultados[categoria] = metricas
            
            # Salvar resultado
            self.salvar_resultado_adios2(categoria, metricas, df_categoria)
        
        # Passo 4: Criar relatório consolidado
        self.criar_relatorio_consolidado(resultados)
        
        print("\n" + "=" * 60)
        print("WORKFLOW CONCLUÍDO COM SUCESSO")
        print("=" * 60)
        print(f"Arquivos salvos em: {self.output_dir}/")
        
    def criar_relatorio_consolidado(self, resultados):
        """
        Cria relatório consolidado de todas as categorias
        """
        print("\n[ADIOS2] Gerando relatório consolidado...")
        
        relatorio_file = os.path.join(self.output_dir, "relatorio_consolidado.csv")
        df_relatorio = pd.DataFrame(list(resultados.values()))
        df_relatorio.to_csv(relatorio_file, index=False)
        
        print(f"[ADIOS2] ✓ Relatório consolidado salvo: {relatorio_file}")


# =====================================================
# ARQUIVO DE CONFIGURAÇÃO ADIOS2 (XML)
# =====================================================

ADIOS2_CONFIG_XML = """<?xml version="1.0"?>
<adios-config>
    
    <!-- Configuração de I/O para leitura de dados -->
    <io name="VendasInput">
        <engine type="BP4">
            <parameter key="Profile" value="On"/>
            <parameter key="ProfileUnits" value="Microseconds"/>
            <parameter key="Threads" value="4"/>
            <parameter key="BufferSize" value="100MB"/>
        </engine>
    </io>
    
    <!-- Configuração de I/O para escrita paralela -->
    <io name="VendasOutput">
        <engine type="BP4">
            <parameter key="Profile" value="On"/>
            <parameter key="Threads" value="4"/>
            <parameter key="BufferSize" value="100MB"/>
            <parameter key="BufferGrowthFactor" value="1.5"/>
            <parameter key="InitialBufferSize" value="50MB"/>
            <parameter key="MaxBufferSize" value="500MB"/>
            <parameter key="FlushStepsCount" value="1"/>
        </engine>
        
        <!-- Variáveis a serem escritas -->
        <variable name="categoria">
            <type>string</type>
        </variable>
        <variable name="produto_id">
            <type>int32</type>
        </variable>
        <variable name="preco">
            <type>double</type>
        </variable>
        <variable name="quantidade">
            <type>int32</type>
        </variable>
        <variable name="data_venda">
            <type>string</type>
        </variable>
        <variable name="media_preco">
            <type>double</type>
        </variable>
        <variable name="desvio_padrao_preco">
            <type>double</type>
        </variable>
        <variable name="total_unidades">
            <type>int64</type>
        </variable>
        <variable name="receita_total">
            <type>double</type>
        </variable>
    </io>
    
    <!-- Configuração de transporte MPI para processamento paralelo -->
    <io name="ParallelProcessing">
        <engine type="BP4">
            <parameter key="OpenTimeoutSecs" value="10"/>
            <parameter key="SubStreams" value="4"/>
            <parameter key="CollectiveMetadata" value="On"/>
        </engine>
    </io>
    
</adios-config>
"""


# =====================================================
# GERADOR DE DATASET DE EXEMPLO
# =====================================================

def gerar_dataset_exemplo(filename="vendas_dataset.csv", n_registros=1000):
    """
    Gera um dataset de exemplo para testar o workflow
    """
    print(f"Gerando dataset de exemplo: {filename}")
    
    np.random.seed(42)
    
    categorias = ['Eletrônicos', 'Roupas', 'Alimentos', 'Livros', 'Móveis']
    
    data = {
        'produto_id': np.arange(1, n_registros + 1),
        'categoria': np.random.choice(categorias, n_registros),
        'preco': np.random.uniform(10, 1000, n_registros).round(2),
        'quantidade': np.random.randint(1, 50, n_registros),
        'data_venda': pd.date_range('2024-01-01', periods=n_registros, freq='H').strftime('%Y-%m-%d')
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
    print(f"✓ Dataset gerado: {n_registros} registros")
    return filename


# =====================================================
# EXECUÇÃO PRINCIPAL
# =====================================================

if __name__ == "__main__":
    # Salvar configuração ADIOS2
    with open("adios2_config.xml", "w") as f:
        f.write(ADIOS2_CONFIG_XML)
    print("✓ Arquivo de configuração ADIOS2 salvo: adios2_config.xml\n")
    
    # Gerar dataset de exemplo
    input_file = gerar_dataset_exemplo()
    
    # Executar workflow
    print()
    workflow = VendasWorkflowADIOS2(input_file)
    workflow.executar_workflow()
    
    print("\n📊 Arquivos gerados:")
    print("  - adios2_config.xml (configuração ADIOS2)")
    print("  - vendas_dataset.csv (dataset de entrada)")
    print("  - output_vendas/ (resultados por categoria)")
    print("  - output_vendas/relatorio_consolidado.csv (resumo geral)")