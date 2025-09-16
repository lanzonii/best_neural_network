class TStudent:
    # Método Construtor
    def __init__(self, a: iter, b: iter):
        '''
        ## Método construtor
        
        Recebe como parâmetros os objetos iteraveis com os valores de cada amostra
        '''
        
        # Sal
        self.a = a
        self.b = b
                
        self.mean_a = self.mean(self.a)
        self.mean_b = self.mean(self.b)
        
        self.s2a = self.s2(self.a)
        self.s2b = self.s2(self.b)
        
        self.df = self.get_df()
        
        self.calculated_t = self.get_calculated_t()
        
        self.t_crit = self.get_t_crit()
    
    # Função que calcula a média dos valores de uma lista
    def mean(self, values: iter):
        return sum(values)/len(values)
    
    # Função que calcula a variância dos valores de uma lista
    def s2(self, values: iter):
        return sum([(x-self.mean(values))**2 for x in values])/(len(values)-1)
    
    def get_df(self):
        nom = ((self.s2a/len(self.a))+(self.s2b/len(self.b)))**2
        denoma = (((self.s2a/len(self.a))**2) / (len(self.a)-1))
        denomb = (((self.s2b/len(self.b))**2)/ (len(self.b)-1))
        
        denom = denoma+denomb
        
        return int(nom/denom)
    
    def get_calculated_t(self):
        nom = self.mean_a-self.mean_b
        denom = (self.s2a/len(self.a))+(self.s2b/len(self.b))
        return nom/(denom**0.5)
    
    def get_t_crit(self):
        import pandas as pd
        
        df = pd.read_csv('t_table.csv')
        df = df.loc[df['df'].astype(int) <= self.df].sort_values(by='df', ascending=False).head(1)
        
        return df["0.05"].iloc[0]
    
    def refuse(self):
        '''
        ## H0: A <= B
        ## H1: A > B
        '''
        return self.calculated_t >= self.t_crit