import os
from pycompss.api.task import task
from pycompss.api.parameter import FILE_OUT, FILE_IN

# Task que gera um número semi-aleatório entre 0 e 32,767
@task(returns=1, output_file=FILE_OUT)
def generate(output_file):
    import random
    number = random.randint(0, 32767)
    with open(output_file, 'w') as f:
        f.write(str(number) + '\n')
    return output_file

# Task que concatena arquivos de entrada em um único arquivo de saída
@task(output_file=FILE_OUT)
def concat(input_files, output_file):
    with open(output_file, 'w') as outf:
        for input_file in input_files:
            with open(input_file, 'r') as inf:
                outf.write(inf.read())

# Task que calcula a soma dos valores em um arquivo de entrada
@task(returns=1, input_file=FILE_IN)
def total(input_file):
    total = 0
    with open(input_file, 'r') as f:
        for line in f:
            total += int(line.strip())
    return total

if __name__ == '__main__':
    from pycompss.api.api import compss_wait_on
    
    # Cria 5 arquivos com números semi-aleatórios em paralelo
    output_files = []
    for i in range(5):
        output_file = os.path.join(os.getcwd(), 'random-{}.txt'.format(i))
        output_files.append(generate(output_file))
    
    # Sincroniza para obter os nomes dos arquivos
    output_files = compss_wait_on(output_files)
    
    # Concatena os arquivos em um único arquivo
    concat_output = os.path.join(os.getcwd(), 'all.txt')
    concat(output_files, concat_output)
    
    # Calcula a soma dos números aleatórios
    result = total(concat_output)
    
    # Sincroniza o resultado final
    result = compss_wait_on(result)
    print(result)