import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
 
scaler = StandardScaler()

class PreProcessamentoBase:
 
    def get_base_titanic():
 
        # Buscando o df
        df_titanic = sns.load_dataset("titanic")
 
        # Removendo colunas que não vão ser usadas
        df_titanic.drop(columns=['alive','deck'], inplace = True)
 
        # Removendo registros vazios baseados na coluna "embark_town"
        df_titanic.dropna(subset=["embark_town"], inplace=True)
 
        # Capturando a média da coluna "age"
        titanic_mediana_age = df_titanic["age"].median()
 
        # trocando os valores vazios na coluna "age" para média da coluna
        df_titanic["age"].fillna(titanic_mediana_age, inplace= True)
 
        # separendo em treino e teste
        X = df_titanic.drop(columns='survived')
        Y = df_titanic['survived']
 
        # Declarando uma lista com as colunas qualitativas
        titanic_colunas_qualitativas = ['sex','class','who','adult_male','embark_town','alone','embarked']
 
        # Iterando as colunas qualittivas e aplicando o fit transform para transformar para quantitativo
        for i in titanic_colunas_qualitativas:
            X[i] = LabelEncoder().fit_transform(X[i])
       
        # Transformar variáveis categóricas em numéricas e guardar encoders
        encoders = {}
        for col in titanic_colunas_qualitativas:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
 
        # Normalizar
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
 
       
       
        retornoBase = [X,Y]
        retornoMetodo = [scaler, encoders]
 
        return retornoBase, retornoMetodo
    
    def transform_titanic(X, transformers):
        scaler, encoders = transformers
        
        X = scaler.transform(X)
        
        for col, le in encoders.items():
            X[col] = le.transform(X[col])
            
        return X
 
    def get_base_cancer_normalized():
 
        # Carregar base
        df_cancer_original = load_breast_cancer()
        df_cancer = pd.DataFrame(df_cancer_original.data, columns=df_cancer_original.feature_names)
        df_cancer['target'] = df_cancer_original.target
 
        # Separar X e y
        X = df_cancer.drop(columns='target')
        Y = df_cancer['target']
 
        # Normalizar features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
 
     
        retornoBase = [X,Y]
        retornoMetodo = [scaler]
       
        return retornoBase, retornoMetodo
    
    def transform_cancer(X, transformers):
        scaler = transformers[0]
        
        X = scaler.transform(X)
            
        return X

    def get_base_adult():
        colunas_adult = [
            'age','workclass','anything','education','education_num',
            'marital_status','occupation','relationship','race','sex',
            'capital_gain', 'capital_loss', 'hours_per_week',
            'native_country', 'income'
        ]
        colunas_adult_Qualitativa = [
            'workclass', 'education','marital_status','occupation',
            'relationship','race','sex', 'native_country'
        ]
 
        # Carregar dataset
        df_adult_original = pd.read_csv('adults.csv', header=None)
        df_adult_original.columns = colunas_adult
 
        # Remover linhas com " ?"
        for i in colunas_adult:
            df_adult_original = df_adult_original.drop(
                df_adult_original[df_adult_original[i] == ' ?'].index
            )
 
        # Separar X e y
        X = df_adult_original.drop(columns='income')
        Y = df_adult_original['income']
 
        # Transformar variáveis categóricas em numéricas e guardar encoders
        encoders = {}
        for col in colunas_adult_Qualitativa:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
 
        # Normalizar
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
 
        # PCA (mantém 95% da variância)
        pca = PCA(n_components=0.95)
        X = pca.fit_transform(X)

        retornoBase = [X,Y]
        retornoMetodo = [scaler, pca, encoders]
       
        return retornoBase, retornoMetodo
    
    def transform_adult(X, transformers):
        scaler, pca, encoders = transformers
        
        X = scaler.transform(X)
        
        X = pca.transform(X)
        
        for col, le in encoders.items():
            X[col] = le.transform(X[col])
            
        return X