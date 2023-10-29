import pandas as pd
import random
import os

loop = 20
dfs = []

def generate_cpf():
    return f"{random.randint(100, 999)}.{random.randint(100, 999)}.{random.randint(100, 999)}-{random.randint(10, 99)}"

def generate_phone():
    return f"({random.randint(10, 99)}) {random.randint(10000, 99999)}-{random.randint(1000, 9999)}"

def generate_address():
    return f"Street {random.choice(['A', 'B', 'C'])}, {random.randint(1, 100)}, City"

def generate_name_surname():
    names = ['Ana', 'João', 'Maria', 'Pedro', 'Lucia', 'Asher', 'José Miguel', 'Miguel', 'José', 'André', 'Henry', 'Isabel', 'Lucas', 'Julia', 'Gabriel', 'Sophia', 'Leonardo', 'Beatriz', 'Arthur', 'Lara', 'Matheus', 'Leticia', 'Carlos', 'Amanda', 'Vinicius', 'Fernanda', 'Bruno', 'Raquel', 'Gustavo', 'Tatiane', 'Rafael', 'Carolina', 'Eduardo', 'Vanessa', 'Diego', 'Camila', 'Alexandre', 'Juliana', 'Victor', 'Priscila', 'Daniel', 'Larissa', 'Fernando', 'Isadora', 'Marco', 'Marina', 'Roberto', 'Bianca', 'Rodrigo', 'Giovanna', 'Ricardo', 'Thais', 'Paulo', 'Natasha', 'Luiz', 'Barbara', 'Guilherme', 'Emily', 'Marcelo', 'Elaine', 'Samuel', 'Aline', 'Antonio', 'Jessica', 'Emanuel', 'Carla', 'Felipe', 'Renata', 'Silvio', 'Adriana', 'Raul', 'Cristiane', 'Adriano', 'Gabriela', 'Alvaro', 'Clarissa', 'Andre', 'Monica', 'Arthur', 'Roberta', 'Augusto', 'Luana', 'Breno', 'Fabiola', 'Caio', 'Laura', 'Cassio', 'Natalia', 'Cesar', 'Andressa', 'Christian', 'Renata', 'Cleber', 'Thalia']

    surnames = ['Silva', 'Santos', 'Oliveira', 'Pereira', 'Lima', 'Da Silva', 'Irineu', 'Josineide', 'Arthur', 'Martins', 'Araujo', 'Gomes', 'Lopes', 'Fernandes', 'Almeida', 'Costa', 'Ribeiro', 'Melo', 'Carvalho', 'Nascimento', 'Dias', 'Cunha', 'Martinez', 'Correa', 'Rocha', 'Pinto', 'Reis', 'Campos', 'Cardoso', 'Castro', 'Cavalcante', 'Duarte', 'Ferreira', 'Garcia', 'Lins', 'Lima', 'Machado', 'Monteiro', 'Morais', 'Nunes', 'Oliveira', 'Pereira', 'Ramos', 'Rezende', 'Rodrigues', 'Santana', 'Santiago', 'Teixeira', 'Vasconcelos', 'Vieira', 'Xavier', 'Zimmermann', 'Vargas', 'Schmidt', 'Ribeiro', 'Pinto', 'Fernandes', 'Barros', 'Goncalves', 'Cardoso', 'Silveira', 'Neves', 'Leal', 'Duarte', 'Ferreira', 'Goncalo', 'Macedo', 'Afonso', 'Guedes', 'Teixeira', 'Borges', 'Pires', 'Pereira', 'Vieira', 'Dantas', 'Cavalcanti', 'Goncalves', 'Pimentel', 'Bittencourt', 'Nogueira', 'Coutinho', 'Souza', 'Cruz', 'Moraes', 'Freitas', 'Correia', 'Andrade', 'Miranda', 'Farias', 'Fonseca', 'Guimaraes', 'Rocha']
    return f"{random.choice(names)} {random.choice(surnames)}"

def generate_cnh():
    return f"{random.randint(10000000000, 99999999999)}"

csv_file = 'ia_reconhecimento_dados/data/training_data.csv'
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    df = pd.DataFrame(columns=['DOCUMENT', 'TYPE'])

for _ in range(100):
    new_data = {
        'DOCUMENT': [generate_cpf(), generate_phone(), generate_address(), generate_name_surname(), generate_cnh()] ,
        'TYPE': ['CPF', 'PHONE', 'ADDRESS', 'NAME_SURNAME', 'CNH'] 
    }
    df_new = pd.DataFrame(new_data)
    dfs.append(df_new)

# Concatenate all dataframes into one
result_df = pd.concat(dfs, ignore_index=True)

df = pd.concat([df, result_df], ignore_index=True)

df.to_csv(csv_file, index=False)
