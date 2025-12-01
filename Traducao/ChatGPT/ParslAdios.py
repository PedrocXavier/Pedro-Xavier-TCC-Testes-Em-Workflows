# import adios2 # Importação incorreta
from adios2 import adios
import os
import random

# ============================================================
# 1. Gerar 5 arquivos ADIOS2 com números aleatórios
# ============================================================

def generate_random_bp_files():
    filenames = []
    for i in range(5):
        value = random.randint(0, 32767)
        fname = f"random-{i}.bp"
        filenames.append(fname)

        # Escrevendo um único inteiro no arquivo BP
        #  with adios2.open(fname, "w") as fh: Linha incorreta original
        with adios.open(fname, "w") as fh:
            fh.write("value", value)

    return filenames

# ============================================================
# 2. Concatenar os valores em um único arquivo BP
# ============================================================

def concatenate_bp_files(input_files, output_file="all.bp"):
    values = []

    # Ler todos os valores dos arquivos BP
    for fname in input_files:
        with adios2.open(fname, "r") as fh:
            for step in fh:
                v = step.read("value")
                values.append(int(v))

    # Escrever o conjunto de valores no arquivo final
    with adios2.open(output_file, "w") as fh:
        fh.write("values", values)

    return output_file

# ============================================================
# 3. Calcular a soma dos valores no arquivo concatenado
# ============================================================

def compute_total_from_bp(bpfile):
    with adios2.open(bpfile, "r") as fh:
        for step in fh:
            values = step.read("values")
            return sum(values)

# ============================================================
# Execução do workflow
# ============================================================

if __name__ == "__main__":
    # 1. Criar arquivos individuais
    random_files = generate_random_bp_files()

    # 2. Concatenar todos
    all_file = concatenate_bp_files(random_files)

    # 3. Somar os valores
    total = compute_total_from_bp(all_file)

    print("Soma dos números aleatórios =", total)
