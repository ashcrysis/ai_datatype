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
import seaborn as sns
import matplotlib.pyplot as plt



def preprocessamento(texto):
    return texto

def carregar_dados_treinamento():
    try:
        dados_treinamento = pd.read_pickle('dados_treinamento_processados.pkl')
    except FileNotFoundError:
        dados_treinamento = pd.read_csv('dados_para_treinamento.csv')
        dados_treinamento.columns = dados_treinamento.columns.str.lower()
        dados_treinamento['documento'] = dados_treinamento['documento'].apply(preprocessamento)
        dados_treinamento.to_pickle('dados_treinamento_processados.pkl')

    return dados_treinamento

def treinar_ou_carregar_modelo():
    try:
        modelo = joblib.load('modelo_treinado.pkl')
        vetorizador = joblib.load('vetorizador.pkl')
    except FileNotFoundError:
        dados_treinamento = carregar_dados_treinamento()
        vetorizador = TfidfVectorizer()
        X = vetorizador.fit_transform(dados_treinamento['documento'])
        y = dados_treinamento['tipo']
        X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
        modelo = SVC(kernel='linear')
        modelo.fit(X_treino, y_treino)
        joblib.dump(modelo, 'modelo_treinado.pkl')
        joblib.dump(vetorizador, 'vetorizador.pkl')
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

exemplo = "Fernanda miguel"

tipo_predito = prever_tipo_documento(exemplo)

print(f"Tipo de documento predito para o dado {exemplo}: {tipo_predito}")
