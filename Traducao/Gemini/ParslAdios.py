import os
import random
from time import sleep

# Importa a biblioteca ADIOS2
# Nota: A instalação do ADIOS2 é necessária para executar este script (pip install adios2)
import adios2

# Função que simula o 'generate' do Parsl.
# Gera um número aleatório e o salva em um arquivo BP do ADIOS2.
def generate_adios(filepath):
    """
    Gera um número semi-aleatório e o escreve em um arquivo ADIOS2 BP.
    """
    # Adios2.open cria um objeto Writer/Reader para o arquivo BP
    # O "r" no modo é uma convenção para o motor BP, mas 'w' é o modo de escrita
    with adios2.open(filepath, "w") as adios_writer:
        # Gera um número aleatório (simula $RANDOM)
        random_value = random.randint(0, 32767)
        
        # Define e escreve a variável 'random_number' no arquivo BP
        adios_writer.write("random_number", [random_value])
        print(f"Gerado {random_value} e salvo em {filepath}")

    # Simula um pequeno atraso para tornar o paralelismo conceitual mais claro
    # Embora em Python, isso será executado sequencialmente.
    sleep(0.01)

# Função que simula o 'total' do Parsl.
# Calcula a soma dos valores lendo múltiplos arquivos ADIOS2 BP.
def total_adios(input_filepaths):
    """
    Lê valores de vários arquivos ADIOS2 BP e calcula a soma total.
    """
    total = 0
    print("\nIniciando cálculo da soma...")

    for filepath in input_filepaths:
        # Adios2.open abre o arquivo BP no modo de leitura ('r')
        with adios2.open(filepath, "r") as adios_reader:
            # Obtém informações sobre a variável que queremos ler
            var_info = adios_reader.inquire_variable("random_number")
            
            if var_info:
                # Cria um array para armazenar o valor
                data = adios_reader.read("random_number")
                # Assumimos que o array tem apenas um elemento, conforme escrito em generate_adios
                value = data[0]
                total += value
                print(f"Lido: {value} de {filepath}")
            else:
                print(f"Aviso: 'random_number' não encontrada em {filepath}")
    
    return total

# --- Workflow Execution (Substituindo a lógica de execução do Parsl) ---

# Configuração de arquivos de saída
output_filepaths = []
num_files = 5
for i in range(num_files):
    output_filepaths.append(os.path.join(os.getcwd(), f'random-{i}.bp')) # Extensão .bp para arquivos ADIOS2

# 1. Create 5 files with semi-random numbers in parallel (Conceitualmente em paralelo, sequencialmente aqui)
# Chamamos generate_adios para cada arquivo.
# No Parsl real, essas chamadas seriam agendadas para execução paralela.
print("--- Gerando arquivos BP ---")
for filepath in output_filepaths:
    generate_adios(filepath)

# 2. Concatenate the files into a single file (PASSO OMITIDO/IMPLÍCITO)
# Não é necessário um passo de concatenação explícita (concat) com ADIOS2. 
# O leitor (total_adios) pode ler diretamente todas as fontes de dados.

# 3. Calculate the sum of the random numbers
print("\n--- Calculando a soma ---")
final_total = total_adios(output_filepaths)

# Resultado final (substitui print(total.result()))
print("\n--- Resultado Final ---")
print(f"Soma Total: {final_total}")

# Opcional: Limpeza dos arquivos gerados (para não poluir o diretório)
# for filepath in output_filepaths:
#     if os.path.exists(filepath):
#         os.remove(filepath)
# print("\nArquivos BP temporários removidos.")