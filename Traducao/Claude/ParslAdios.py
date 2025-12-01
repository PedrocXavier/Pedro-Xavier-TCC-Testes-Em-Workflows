import adios2
import numpy as np
import random
import os
from mpi4py import MPI

# Inicializa MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Configuração do ADIOS 2
adios = adios2.Adios(comm)

# Etapa 1: Gerar números semi-aleatórios em paralelo
def generate_numbers(num_files=5):
    """Gera números semi-aleatórios e escreve usando ADIOS 2"""
    
    # Cada processo gera seus números
    numbers = []
    for i in range(num_files):
        if i % size == rank:  # Distribuição de trabalho entre processos
            random_num = random.randint(0, 32767)
            numbers.append((i, random_num))
            print(f"Rank {rank}: Gerou número {random_num} para arquivo {i}")
    
    # Escreve os números usando ADIOS 2
    io = adios.DeclareIO("NumberWriter")
    io.SetEngine("BP4")  # Engine BP4 para arquivos binários
    
    writer = io.Open("random_numbers.bp", adios2.Mode.Write)
    
    for idx, num in numbers:
        var = io.DefineVariable(f"number_{idx}", 
                                np.array([num]), 
                                [1], [0], [1])
        writer.Put(var, np.array([num]))
    
    writer.Close()
    
    # Sincroniza todos os processos
    comm.Barrier()
    
    return num_files

# Etapa 2: Concatenar/Ler todos os números
def concat_numbers(num_files):
    """Lê todos os números usando ADIOS 2"""
    
    all_numbers = []
    
    if rank == 0:  # Apenas o processo mestre faz a leitura
        io = adios.DeclareIO("NumberReader")
        reader = io.Open("random_numbers.bp", adios2.Mode.Read)
        
        for i in range(num_files):
            var = io.InquireVariable(f"number_{i}")
            if var:
                data = np.zeros(1, dtype=int)
                reader.Get(var, data)
                reader.PerformGets()
                all_numbers.append(data[0])
                print(f"Lido número: {data[0]}")
        
        reader.Close()
        
        # Escreve todos os números concatenados
        io_concat = adios.DeclareIO("ConcatWriter")
        io_concat.SetEngine("BP4")
        writer = io_concat.Open("all_numbers.bp", adios2.Mode.Write)
        
        var_all = io_concat.DefineVariable("all_numbers",
                                           np.array(all_numbers),
                                           [len(all_numbers)],
                                           [0],
                                           [len(all_numbers)])
        writer.Put(var_all, np.array(all_numbers))
        writer.Close()
    
    comm.Barrier()
    return all_numbers

# Etapa 3: Calcular o total
def calculate_total():
    """Calcula a soma de todos os números"""
    
    total = 0
    
    if rank == 0:
        io = adios.DeclareIO("TotalReader")
        reader = io.Open("all_numbers.bp", adios2.Mode.Read)
        
        var = io.InquireVariable("all_numbers")
        if var:
            shape = var.Shape()
            data = np.zeros(shape[0], dtype=int)
            reader.Get(var, data)
            reader.PerformGets()
            total = np.sum(data)
            print(f"\nNúmeros concatenados: {data}")
        
        reader.Close()
    
    # Broadcast do total para todos os processos
    total = comm.bcast(total, root=0)
    
    return total

# Execução do workflow
if __name__ == "__main__":
    print(f"Iniciando workflow no rank {rank} de {size}")
    
    # Gera números
    num_files = generate_numbers(5)
    
    # Concatena números
    concat_numbers(num_files)
    
    # Calcula total
    total = calculate_total()
    
    if rank == 0:
        print(f"\nTotal: {total}")
    
    # Finaliza ADIOS
    adios.RemoveAllIOs()