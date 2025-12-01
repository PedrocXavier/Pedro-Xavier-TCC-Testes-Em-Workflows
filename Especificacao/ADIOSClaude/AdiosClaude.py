#!/usr/bin/env python3
"""
Workflow ADIOS2 para análise paralela de dados de vendas no varejo.
Agrupa por categoria e calcula estatísticas em paralelo.
"""

import adios2
import numpy as np
import pandas as pd
from mpi4py import MPI
import sys
import os

class VendasWorkflow:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.adios = adios2.Adios(self.comm)
        
    def carregar_dados(self, arquivo_csv):
        """Carrega o dataset de vendas (apenas no rank 0)"""
        if self.rank == 0:
            print(f"[Rank {self.rank}] Carregando dados de {arquivo_csv}...")
            df = pd.read_csv(arquivo_csv)
            print(f"[Rank {self.rank}] Dados carregados: {len(df)} registros")
            return df
        return None
    
    def distribuir_categorias(self, df):
        """Distribui categorias entre os processos MPI"""
        if self.rank == 0:
            categorias = df['categoria'].unique()
            print(f"[Rank {self.rank}] Categorias encontradas: {list(categorias)}")
            
            # Distribuir categorias entre ranks
            categorias_por_rank = np.array_split(categorias, self.size)
        else:
            df = None
            categorias_por_rank = None
        
        # Broadcast do dataframe completo para todos os ranks
        df = self.comm.bcast(df, root=0)
        
        # Scatter das categorias
        minhas_categorias = self.comm.scatter(categorias_por_rank, root=0)
        
        if self.rank == 0:
            print(f"\n[Rank {self.rank}] Distribuição de categorias:")
            for i, cats in enumerate(categorias_por_rank):
                print(f"  Rank {i}: {list(cats)}")
        
        return df, minhas_categorias
    
    def calcular_estatisticas(self, df, categoria):
        """Calcula estatísticas para uma categoria específica"""
        df_cat = df[df['categoria'] == categoria].copy()
        
        # Calcular métricas
        stats = {
            'categoria': categoria,
            'media_preco': df_cat['preco'].mean(),
            'desvio_preco': df_cat['preco'].std(),
            'total_unidades': df_cat['quantidade'].sum(),
            'receita_total': (df_cat['preco'] * df_cat['quantidade']).sum(),
            'num_vendas': len(df_cat)
        }
        
        print(f"[Rank {self.rank}] Categoria '{categoria}':")
        print(f"  - Média de preço: R$ {stats['media_preco']:.2f}")
        print(f"  - Desvio padrão: R$ {stats['desvio_preco']:.2f}")
        print(f"  - Total de unidades: {stats['total_unidades']}")
        print(f"  - Receita total: R$ {stats['receita_total']:.2f}")
        
        return stats, df_cat
    
    def processar_minhas_categorias(self, df, categorias):
        """Processa todas as categorias atribuídas a este rank"""
        resultados = []
        dataframes = []
        
        for categoria in categorias:
            stats, df_cat = self.calcular_estatisticas(df, categoria)
            resultados.append(stats)
            dataframes.append(df_cat)
        
        return resultados, dataframes
    
    def escrever_com_adios(self, resultados, dataframes):
        """Escreve os resultados usando ADIOS2"""
        io = self.adios.declare_io("VendasIO")
        io.set_engine("BP4")  # Engine BP4 para I/O paralelo eficiente
        
        # Criar diretório de saída
        if self.rank == 0:
            os.makedirs("output_adios", exist_ok=True)
        self.comm.Barrier()
        
        # Escrever arquivo BP com todas as estatísticas
        writer = io.open("output_adios/vendas_stats.bp", adios2.Mode.Write)
        
        for stats in resultados:
            # Definir variáveis ADIOS2 para cada métrica
            var_cat = io.define_variable(f"{stats['categoria']}_nome")
            var_media = io.define_variable(f"{stats['categoria']}_media_preco")
            var_desvio = io.define_variable(f"{stats['categoria']}_desvio_preco")
            var_unidades = io.define_variable(f"{stats['categoria']}_total_unidades")
            var_receita = io.define_variable(f"{stats['categoria']}_receita_total")
            var_num = io.define_variable(f"{stats['categoria']}_num_vendas")
            
            # Escrever dados
            writer.put(var_cat, stats['categoria'])
            writer.put(var_media, np.array([stats['media_preco']]))
            writer.put(var_desvio, np.array([stats['desvio_preco']]))
            writer.put(var_unidades, np.array([stats['total_unidades']]))
            writer.put(var_receita, np.array([stats['receita_total']]))
            writer.put(var_num, np.array([stats['num_vendas']]))
        
        writer.close()
        
        if self.rank == 0:
            print(f"\n[Rank {self.rank}] Estatísticas escritas em output_adios/vendas_stats.bp")
    
    def salvar_csvs(self, resultados, dataframes):
        """Salva cada categoria como CSV separado"""
        if self.rank == 0:
            os.makedirs("output_csv", exist_ok=True)
        self.comm.Barrier()
        
        for stats, df_cat in zip(resultados, dataframes):
            categoria = stats['categoria']
            
            # Criar dataframe com estatísticas
            df_stats = pd.DataFrame([stats])
            
            # Salvar estatísticas
            stats_file = f"output_csv/{categoria}_estatisticas.csv"
            df_stats.to_csv(stats_file, index=False)
            
            # Salvar dados detalhados
            dados_file = f"output_csv/{categoria}_dados.csv"
            df_cat.to_csv(dados_file, index=False)
            
            print(f"[Rank {self.rank}] Salvos arquivos para categoria '{categoria}':")
            print(f"  - {stats_file}")
            print(f"  - {dados_file}")
    
    def executar(self, arquivo_csv):
        """Executa o workflow completo"""
        print(f"\n{'='*60}")
        print(f"WORKFLOW ADIOS2 - ANÁLISE DE VENDAS")
        print(f"Rank {self.rank} de {self.size}")
        print(f"{'='*60}\n")
        
        # 1. Carregar dados
        df = self.carregar_dados(arquivo_csv)
        
        # 2. Distribuir categorias entre processos
        df, minhas_categorias = self.distribuir_categorias(df)
        
        # 3. Processar categorias em paralelo
        print(f"\n[Rank {self.rank}] Processando {len(minhas_categorias)} categoria(s)...\n")
        resultados, dataframes = self.processar_minhas_categorias(df, minhas_categorias)
        
        # Sincronizar antes de escrever
        self.comm.Barrier()
        
        # 4. Escrever resultados com ADIOS2
        if len(resultados) > 0:
            self.escrever_com_adios(resultados, dataframes)
        
        # 5. Salvar CSVs individuais
        if len(resultados) > 0:
            self.salvar_csvs(resultados, dataframes)
        
        # Sincronizar no final
        self.comm.Barrier()
        
        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"WORKFLOW CONCLUÍDO COM SUCESSO!")
            print(f"{'='*60}\n")


def main():
    if len(sys.argv) < 2:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("Uso: mpirun -np <num_processos> python workflow_vendas.py <arquivo_csv>")
            print("Exemplo: mpirun -np 4 python workflow_vendas.py vendas.csv")
        sys.exit(1)
    
    arquivo_csv = sys.argv[1]
    
    # Validar arquivo apenas no rank 0
    if MPI.COMM_WORLD.Get_rank() == 0:
        if not os.path.exists(arquivo_csv):
            print(f"Erro: Arquivo '{arquivo_csv}' não encontrado!")
            MPI.COMM_WORLD.Abort(1)
    
    # Executar workflow
    workflow = VendasWorkflow()
    workflow.executar(arquivo_csv)


if __name__ == "__main__":
    main()