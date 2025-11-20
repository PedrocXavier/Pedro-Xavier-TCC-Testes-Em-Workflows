import os
import random

from pycompss.api.task import task
from pycompss.api.parameter import FILE_IN, FILE_OUT, COLLECTION_IN, COLLECTION_OUT
from pycompss.api.api import compss_wait_on


# ----------------------------------------------------------------------
# Gera um número aleatório e grava em um arquivo
# ----------------------------------------------------------------------
@task(file_path=FILE_OUT)
def generate(file_path):
    with open(file_path, "w") as f:
        value = random.randint(0, 32767)
        f.write(str(value) + "\n")


# ----------------------------------------------------------------------
# Concatena vários arquivos em um único arquivo
# ----------------------------------------------------------------------
@task(inputs=COLLECTION_IN, output_file=FILE_OUT)
def concat(inputs, output_file):
    with open(output_file, "w") as out:
        for fname in inputs:
            with open(fname, "r") as f:
                out.write(f.read())


# ----------------------------------------------------------------------
# Soma os valores contidos em um arquivo
# ----------------------------------------------------------------------
@task(file_path=FILE_IN, returns=int)
def total(file_path):
    s = 0
    with open(file_path, "r") as f:
        for line in f:
            s += int(line)
    return s


# ======================================================================
# MAIN WORKFLOW
# ======================================================================
def main():
    cwd = os.getcwd()

    # 1. Gerar 5 arquivos com números aleatórios (em paralelo)
    output_files = []
    for i in range(5):
        fpath = os.path.join(cwd, f"random-{i}.txt")
        generate(fpath)
        output_files.append(fpath)

    # 2. Concatenar arquivos em all.txt
    all_file = os.path.join(cwd, "all.txt")
    concat(output_files, all_file)

    # 3. Calcular a soma dos números do arquivo concatenado
    result = total(all_file)

    # Esperar o resultado final
    result = compss_wait_on(result)
    print("Soma total =", result)


if __name__ == "__main__":
    main()
