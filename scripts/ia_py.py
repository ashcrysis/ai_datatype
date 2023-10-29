import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import re
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def preprocessamento(texto):
    if isinstance(texto, str):
        return texto.lower()
    elif isinstance(texto, float):
        return str(texto).lower()
    else:
        return str(texto)

def carregar_dados_treinamento():
    try:
        dados_treinamento = pd.read_pickle('ia_reconhecimento_dados/cache/processed_data.pkl')
    except FileNotFoundError:
        dados_treinamento = pd.read_csv('ia_reconhecimento_dados/data/training_data.csv')
        dados_treinamento.columns = [preprocessamento(col) for col in dados_treinamento.columns]
        dados_treinamento['document'] = dados_treinamento['document'].apply(preprocessamento)
        dados_treinamento.to_pickle('ia_reconhecimento_dados/cache/processed_data.pkl')

    return dados_treinamento

def treinar_ou_carregar_modelo():
    try:
        modelo = joblib.load('ia_reconhecimento_dados/cache/modelo_treinado.pkl')
        vetorizador = joblib.load('ia_reconhecimento_dados/cache/vetorizador.pkl')
    except FileNotFoundError:
        dados_treinamento = carregar_dados_treinamento()
        vetorizador = TfidfVectorizer()
        X = vetorizador.fit_transform(dados_treinamento['document'])
        y = dados_treinamento['type']
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo = SVC(kernel='linear')
        modelo.fit(X_treino, y_treino)
        joblib.dump(modelo, 'ia_reconhecimento_dados/cache/modelo_treinado.pkl')
        joblib.dump(vetorizador, 'ia_reconhecimento_dados/cache/vetorizador.pkl')
        previsoes = modelo.predict(X_teste)

        precisao = accuracy_score(y_teste, previsoes)
        print(f'Acurácia do modelo: {precisao}')
        matriz_confusao = confusion_matrix(y_teste, previsoes)
        sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=modelo.classes_, yticklabels=modelo.classes_)
        plt.xlabel('Previsto')
        plt.ylabel('Real')
        plt.show()
    return modelo, vetorizador

modelo, vetorizador = treinar_ou_carregar_modelo()

def prever_tipo_documento(documento):
    documento = preprocessamento(documento)
    documento_vetorizado = vetorizador.transform([documento])
    predicao = modelo.predict(documento_vetorizado)[0]
    return predicao

def avaliar_predicao(input_text,tipo_real, tipo_predito):
    resposta = input(f"O tipo predito '{tipo_predito}' para o documento está correto? (s/n): ").lower()
    if resposta == 's':
        correto = True
    elif resposta == 'n':
        correto = False
    else:
        print("Resposta inválida. Por favor, responda 's' para correto ou 'n' para incorreto.")
        correto = avaliar_predicao( input_text,tipo_real, tipo_predito)

    with open('ia_reconhecimento_dados/data/feedback.csv', 'a') as file:
        file.write(f"Text:{input_text}, Text_Real_Type:{tipo_real},\nPredicted_Type:{tipo_predito}, Predction_Status:{correto}\n")

    return correto

exemplo = "(81) 98434-9230"
tipo_real = "PHONE"  

tipo_predito = prever_tipo_documento(exemplo)
print("Exemplo inserido: " , exemplo)
print("Tipo Real: ", tipo_real)
correto = avaliar_predicao(exemplo ,tipo_real, tipo_predito)
if correto:
    print("A IA acertou!")
else:
    print("A IA errou. Vamos melhorar o treinamento.")
