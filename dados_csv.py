import pandas as pd
import random
import os

def gerar_cpf():
    return f"{random.randint(100, 999)}.{random.randint(100, 999)}.{random.randint(100, 999)}-{random.randint(10, 99)}"

def gerar_telefone():
    return f"({random.randint(10, 99)}) {random.randint(10000, 99999)}-{random.randint(1000, 9999)}"

def gerar_endereco():
    return f"Rua {random.choice(['A', 'B', 'C'])}, {random.randint(1, 100)}, Cidade"

def gerar_nome_sobrenome():
    nomes = ['Ana', 'João', 'Maria', 'Pedro', 'Lucia', 'Asher', 'José Miguel', 'Miguel', 'José', 'André', 'Henry', 'Isabel', 'Lucas', 'Julia', 'Gabriel', 'Sophia', 'Leonardo', 'Beatriz', 'Arthur', 'Lara', 'Matheus', 'Leticia', 'Carlos', 'Amanda', 'Vinicius', 'Fernanda', 'Bruno', 'Raquel', 'Gustavo', 'Tatiane', 'Rafael', 'Carolina', 'Eduardo', 'Vanessa', 'Diego', 'Camila', 'Alexandre', 'Juliana', 'Victor', 'Priscila', 'Daniel', 'Larissa', 'Fernando', 'Isadora', 'Marco', 'Marina', 'Roberto', 'Bianca', 'Rodrigo', 'Giovanna', 'Ricardo', 'Thais', 'Paulo', 'Natasha', 'Luiz', 'Barbara', 'Guilherme', 'Emily', 'Marcelo', 'Elaine', 'Samuel', 'Aline', 'Antonio', 'Jessica', 'Emanuel', 'Carla', 'Felipe', 'Renata', 'Silvio', 'Adriana', 'Raul', 'Cristiane', 'Adriano', 'Gabriela', 'Alvaro', 'Clarissa', 'Andre', 'Monica', 'Arthur', 'Roberta', 'Augusto', 'Luana', 'Breno', 'Fabiola', 'Caio', 'Laura', 'Cassio', 'Natalia', 'Cesar', 'Andressa', 'Christian', 'Renata', 'Cleber', 'Thalia']

    sobrenomes = ['Silva', 'Santos', 'Oliveira', 'Pereira', 'Lima', 'Da Silva', 'Irineu', 'Josineide', 'Arthur', 'Martins', 'Araujo', 'Gomes', 'Lopes', 'Fernandes', 'Almeida', 'Costa', 'Ribeiro', 'Melo', 'Carvalho', 'Nascimento', 'Dias', 'Cunha', 'Martinez', 'Correa', 'Rocha', 'Pinto', 'Reis', 'Campos', 'Cardoso', 'Castro', 'Cavalcante', 'Duarte', 'Ferreira', 'Garcia', 'Lins', 'Lima', 'Machado', 'Monteiro', 'Morais', 'Nunes', 'Oliveira', 'Pereira', 'Ramos', 'Rezende', 'Rodrigues', 'Santana', 'Santiago', 'Teixeira', 'Vasconcelos', 'Vieira', 'Xavier', 'Zimmermann', 'Vargas', 'Schmidt', 'Ribeiro', 'Pinto', 'Fernandes', 'Barros', 'Goncalves', 'Cardoso', 'Silveira', 'Neves', 'Leal', 'Duarte', 'Ferreira', 'Goncalo', 'Macedo', 'Afonso', 'Guedes', 'Teixeira', 'Borges', 'Pires', 'Pereira', 'Vieira', 'Dantas', 'Cavalcanti', 'Goncalves', 'Pimentel', 'Bittencourt', 'Nogueira', 'Coutinho', 'Souza', 'Cruz', 'Moraes', 'Freitas', 'Correia', 'Andrade', 'Miranda', 'Farias', 'Fonseca', 'Guimaraes', 'Rocha']
    return f"{random.choice(nomes)} {random.choice(sobrenomes)}"

def gerar_cnh():
    return f"{random.randint(10000000000, 99999999999)}"

arquivo_csv = 'dados_para_treinamento.csv'
if os.path.exists(arquivo_csv):
    df = pd.read_csv(arquivo_csv)
else:
    df = pd.DataFrame(columns=['DOCUMENTO', 'TIPO'])

novos_dados = {
    'DOCUMENTO': [gerar_cpf(), gerar_telefone(), gerar_endereco(), gerar_nome_sobrenome(), gerar_cnh()] * 20,
    'TIPO': ['CPF', 'TELEFONE', 'ENDEREÇO', 'NOME_SOBRENOME', 'CNH'] * 20
}
df_novos = pd.DataFrame(novos_dados)

df = pd.concat([df, df_novos], ignore_index=True)

df.to_csv(arquivo_csv, index=False)