import os
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN, FILE_OUT, STDOUT
import random

# App (Task) que gera um número semi-aleatório e o escreve em um arquivo de saída
# Usamos uma tarefa Python simples para evitar a complexidade de envolver comandos bash externos 
# para uma funcionalidade tão trivial, seguindo as melhores práticas do PyCOMPSs.
@task(returns=FILE_OUT)
def generate():
    """Gera um número semi-aleatório entre 0 e 32767 e o salva em um arquivo."""
    # O PyCOMPSs cria o nome do arquivo temporário/de saída quando FILE_OUT é usado
    # Retornamos apenas o conteúdo que o PyCOMPSs salvará no arquivo
    return random.randint(0, 32767)

# App (Task) que concatena arquivos de entrada em um único arquivo de saída
@task(inputs={**args: FILE_IN for args in ["input_files"]}, target=FILE_OUT)
def concat(input_files):
    """Concatena o conteúdo dos arquivos de entrada em um arquivo de saída."""
    output_filename = os.environ.get("COMPSs_FILENAME") # PyCOMPSs define esta variável
    
    with open(output_filename, 'w') as outfile:
        for fname in input_files:
            with open(fname, 'r') as infile:
                outfile.write(infile.read())
    # Em tarefas com target=FILE_OUT, não precisamos retornar nada, 
    # pois o PyCOMPSs gerencia a escrita no arquivo de saída (output_filename)
    return True # Retorno de sucesso (opcional)

# App (Task) que calcula a soma dos valores em um único arquivo de entrada
@task(file_in=FILE_IN, returns=int)
def total(file_in):
    """Calcula a soma dos valores contidos no arquivo de entrada."""
    total_sum = 0
    with open(file_in, 'r') as f:
        # A implementação original do Parsl parecia esperar que cada linha fosse um número
        for l in f:
            try:
                # Remove espaços em branco (incluindo quebras de linha) e converte para inteiro
                total_sum += int(l.strip())
            except ValueError:
                # Trata a possibilidade de linhas vazias ou não numéricas se necessário
                pass 
    return total_sum

# --- Workflow Principal ---
if __name__ == '__main__':
    # 1. Cria 5 arquivos com números semi-aleatórios em paralelo
    # O PyCOMPSs gerencia os nomes dos arquivos automaticamente. 
    # As chamadas retornam Futuros (Future Objects).
    output_futures = []
    for i in range(5):
        # A chamada da função generate() retorna um Future que representa o arquivo de saída
        output_futures.append(generate())

    # 2. Concatena os arquivos em um único arquivo
    # Passamos a lista de Futuros de arquivo como argumento.
    # O PyCOMPSs garante que a tarefa concat só comece após todos os Futuros estarem prontos.
    # A chamada retorna um Future que representa o arquivo de saída concatenado.
    cc_future = concat(output_futures)

    # 3. Calcula a soma dos números aleatórios
    # Passamos o Future do arquivo concatenado para a tarefa total.
    # A chamada retorna um Future para o resultado inteiro da soma.
    total_future = total(cc_future)

    # 4. Espera pelo resultado final e o imprime
    # compss_wait_on é o equivalente a .result() do Parsl, 
    # que bloqueia até que o Future Object seja resolvido.
    final_total = compss_wait_on(total_future)
    
    print(f"Soma Total: {final_total}")