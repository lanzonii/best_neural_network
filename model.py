import tensorflow as tf 
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from t_student import TStudent
import plotly.express as px
from pre_processamento_bases import PreProcessamentoBase
import random

class ModelTraining:
    
    def __init__(self, x, y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    
    def best_activation(self, model: Sequential, neuron_amount, neurons):
        if neurons > 0:
            original_model = clone_model(model)
        activation_accuracies = {}
        activation_models = {}

        for activation in ['sigmoid', 'softplus', 'relu', 'tanh']:
            if neurons > 0:
                model = clone_model(original_model)
            else:
                model = Sequential()
            
            model.add(Dense(neuron_amount, activation=activation, kernel_initializer=GlorotUniform(seed=42)))
            
            model.add(Dense(1, activation='sigmoid', kernel_initializer=GlorotUniform(seed=42)))
            
            untrained = model

            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            history = model.fit(self.x_train, self.y_train, epochs=self.epochs, validation_data=(self.x_test, self.y_test), verbose=0)

            loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
            
            val_accuracy = history.history['val_accuracy']

            activation_models[activation] = {'history': history.history, 'accuracy': accuracy, 'model': model, 'val_accuracy': val_accuracy, 'untrained': untrained}
            activation_accuracies[activation] = accuracy

            print('     ', activation)
            print('          accuracy: ', accuracy)

            activation_accuracies[activation] = accuracy
            
        best_activation = max(activation_accuracies, key=activation_accuracies.get)
        
        print('\n     best_activation: ', best_activation)

        return activation_models[best_activation], best_activation

    def best_optimizer(self, model):
        original_model = clone_model(model)
        optimizer_models = {}
        optimizer_accuracies = {}
        
        for optimizer in ['adam', 'sgd', 'nadam', 'adamax', 'adagrad', 'rmsprop']:
            model = original_model

            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
            
            history = model.fit(self.x_train, self.y_train, epochs=self.epochs, validation_data=(self.x_test, self.y_test), verbose=0)

            loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)

            val_accuracy = history.history['val_accuracy']

            optimizer_models[optimizer] = {'history': history.history, 'accuracy': accuracy, 'model': model, 'val_accuracy': val_accuracy}
            optimizer_accuracies[optimizer] = accuracy
            
        best_optimizer = max(optimizer_accuracies, key=optimizer_accuracies.get)
        
        return optimizer_models[best_optimizer]
                    

    def train(self, epochs):
        self.epochs = epochs
        
        accuracies = []
        neurons = []
        density = 1
        best_model_density = {'accuracy': 0}
        big_same = 0
        
        loops = 1
        
        # Loop para adicionar camadas
        while True:
            
            best_model = {'accuracy': 0}
            
            size = 1
            
            same = 0
            # Loop para adicionar neurônios na mesma camada
            while True:
                print(f'\n{density} camada(s), {size} neurônio(s):')
                
                model = Sequential()

                if len(neurons) > 0:
                    for neuron in neurons:
                        model.add(neuron)
                
                model, activation = self.best_activation(model, size, len(neurons))

                accuracies.append(model['accuracy'])
                
                if same < 3:
                    if model['accuracy'] > best_model['accuracy']:
                        best_model = model
                        layer = Dense(size, activation=activation, kernel_initializer=GlorotUniform(seed=42))
                        
                        if len(neurons) < density:
                            neurons.append(layer)   # primeira vez, adiciona
                        else:
                            neurons[density-1] = layer        
                
                        size += 1
                        same = 0
                    else:
                        same+=1
                else:
                    break
                
                loops += 1
            
            if big_same < 3:
                if best_model['accuracy'] > best_model_density['accuracy']:
                    best_model_density = best_model
                    
                    density+=1
                    big_same = 0
                else:
                    big_same += 1
            else:
                break
        px.line(y=accuracies).show()
        
        self.best_model = self.best_optimizer(best_model_density['untrained'])
        
        return self
