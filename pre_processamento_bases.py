import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MaxAbsScaler



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

        # Declarando uma lista com as colunas qualitativas
        titanic_colunas_qualitativas = ['sex','class','who','adult_male','embark_town','alone','embarked']

        # Iterando as colunas qualittivas e aplicando o fit transform para transformar para quantitativo
        for i in titanic_colunas_qualitativas:
            df_titanic[i] = LabelEncoder().fit_transform(df_titanic[i])
        
        x = df_titanic.drop(columns='survived')
        y = df_titanic['survived']

        return x,y

    def get_base_cancer():

        
        # Puxando base
        df_cancer_original = load_breast_cancer()
         
        # Criar DataFrame com as colunas
        df_cancer = pd.DataFrame(df_cancer_original.data, columns=df_cancer_original.feature_names)

        # Adicionar a coluna Y (por algum motivo fica separado)
        df_cancer["target"] = pd.DataFrame(df_cancer_original.target)
        
        x = df_cancer.drop(columns='target')
        y = df_cancer['target']

        return x,y
    
    def get_base_cancer_normalized():
        df_cancer_original = load_breast_cancer()

        df_cancer = pd.DataFrame(df_cancer_original.data, columns=df_cancer_original.feature_names)

        df_cancer["target"] = pd.DataFrame(df_cancer_original.target)

        columns_cancer = df_cancer.columns

        scaler = StandardScaler()

        df_cancer_normalized = scaler.fit_transform(df_cancer)

        df_cancer_normalized = pd.DataFrame(df_cancer_normalized, columns=columns_cancer)

        x = df_cancer_normalized.drop(columns='target')
        y = df_cancer_normalized['target']

        return x,y

    def get_base_adult():

        colunas_adult = ['age','workclass','anything','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

        df_adult_original = pd.read_csv('adults.csv', header=None)

        df_adult_original.columns = colunas_adult

        for i in colunas_adult:
            df_adult = df_adult_original.drop(df_adult_original[df_adult_original[i] == ' ?'].index)

        for i in colunas_adult:
            df_adult[i] = LabelEncoder().fit_transform(df_adult[i])
 
        x = df_adult.drop(columns='income')
        y = df_adult['income']
        
        return x, y