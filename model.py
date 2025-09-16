import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import GlorotUniform
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from t_student import TStudent
import plotly.express as px
from pre_processamento_bases import PreProcessamentoBase
import random

class ModelTraining:
    
    def __init__(self, x, y):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    
    def best_activation(self, model, neuron_amount):
        original_model = model
        activation_accuracies = {}
        
        for activation in ['sigmoid', 'softplus', 'relu', 'tanh']:
            model = original_model
            
            model.add(Dense(neuron_amount, activation=activation, kernel_initializer=GlorotUniform(seed=42)))
            
            model.add(Dense(1, activation='sigmoid', kernel_initializer=GlorotUniform(seed=42)))
            
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            history = model.fit(self.x_train, self.y_train, epochs=self.epochs, validation_data=(self.x_test, self.y_test), verbose=0)

            loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)

            activation_accuracies[activation] = accuracy
            
        best_activation = max(activation_accuracies, key=activation_accuracies.get)
        
        return best_activation

    def best_optimizer(self, model):
        original_model = model
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
        
        return best_optimizer, optimizer_models[best_optimizer]
                    

    def train(self, epochs):
        self.epochs = epochs
        
        accuracies = []
        neurons = []
        density = 1
        best_model_density = {'history': {'val_accuracy': [0, 0, 0]}}
        
        # Loop para adicionar camadas
        while True:
            
            best_model = {'history': {'val_accuracy': [0, 0, 0]}}
            
            size = 1
            # Loop para adicionar neurônios na mesma camada
            while True:
                
                model = Sequential()

                if len(neurons) > 0:
                    for neuron in neurons:
                        model.add(neuron)
                
                activation = self.best_activation(model, size)
                model.add(Dense(size, activation=activation, kernel_initializer=GlorotUniform(seed=42)))
                
                model.add(Dense(1, activation='sigmoid', kernel_initializer=GlorotUniform(seed=42)))
                
                untrained_model = model
                
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

                history = model.fit(self.x_train, self.y_train, epochs=self.epochs, validation_data=(self.x_test, self.y_test), verbose=0)

                loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)

                val_accuracy = history.history['val_accuracy']

                accuracies.append(accuracy)
                
                if TStudent(val_accuracy, best_model['history']['val_accuracy']).refuse():
                    best_model['history'] = history.history
                    best_model['model'] = model
                    best_model['untrained'] = untrained_model
                    neurons[density-1] = Dense(size, activation=activation, kernel_initializer=GlorotUniform(seed=42))
                    
                    size += 1
                else:
                    break
            
            if TStudent(best_model['history']['val_accuracy'], best_model_density['history']['val_accuracy']).refuse():
                best_model_density = best_model
                
                density+=1
            else:
                break
        
        px.line(y=accuracies).show()
        
        self.best_model = self.best_optimizer(best_model_density['untrained'])[['history', 'model']]
        
        return self
